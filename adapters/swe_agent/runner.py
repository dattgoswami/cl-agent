from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from .context_builder import ContextBuilder, SWEAgentRunContext


@dataclass(frozen=True)
class SWEAgentProcessResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SWEAgentCommandPreview:
    run_id: str
    context: SWEAgentRunContext
    args: list[str]
    config_path: str
    problem_statement_path: str
    output_dir: str | None


@dataclass(frozen=True)
class SWEAgentRunResult:
    preview: SWEAgentCommandPreview
    process: SWEAgentProcessResult


CommandRunner = Callable[
    [Sequence[str], str | None, Mapping[str, str] | None, int | None],
    SWEAgentProcessResult,
]


class SWEAgentRunner:
    """
    Thin, injectable SWE-agent command boundary.

    The adapter is import-first. This runner writes an inspectable CL config
    overlay and problem-statement file, builds the `sweagent run` command, and
    delegates execution to an injected command runner. Tests can exercise the
    command/config boundary without Docker, API keys, network access, or
    benchmark downloads.
    """

    def __init__(
        self,
        *,
        artifacts_dir: str | Path | None = None,
        run_artifacts_dir: str | Path | None = None,
        sweagent_executable: str = "sweagent",
        base_config: str | Path | None = None,
        timeout_seconds: int | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.context_builder = ContextBuilder(artifacts_dir)
        self.run_artifacts_dir = Path(run_artifacts_dir) if run_artifacts_dir else Path("swe-agent-captures")
        self.sweagent_executable = sweagent_executable
        self.base_config = str(base_config) if base_config is not None else None
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner or self._unconfigured_runner

    def build_command(
        self,
        *,
        config_path: str,
        problem_statement_path: str,
        task_id: str,
        repo_path: str | None = None,
        github_url: str | None = None,
        model: str | None = None,
        output_dir: str | None = None,
        extra_args: Sequence[str] = (),
    ) -> list[str]:
        args = [self.sweagent_executable, "run"]
        if self.base_config:
            args.extend(["--config", self.base_config])
        args.extend(["--config", config_path])
        args.extend(
            [
                "--problem_statement.type=text_file",
                f"--problem_statement.path={problem_statement_path}",
                f"--problem_statement.id={task_id}",
            ]
        )
        if repo_path:
            args.extend(["--env.repo.type=local", f"--env.repo.path={repo_path}"])
        if github_url:
            args.extend(["--env.repo.type=github", f"--env.repo.github_url={github_url}"])
        if model:
            args.append(f"--agent.model.name={model}")
        if output_dir:
            args.append(f"--output_dir={output_dir}")
        args.extend(extra_args)
        return args

    def preview(
        self,
        task_prompt: str,
        *,
        task_id: str,
        mode: Literal["baseline", "integrated"] = "baseline",
        cwd: str | None = None,
        repo_path: str | None = None,
        github_url: str | None = None,
        model: str | None = None,
        output_dir: str | Path | None = None,
        extra_args: Sequence[str] = (),
    ) -> SWEAgentCommandPreview:
        if repo_path and github_url:
            raise ValueError("Provide either repo_path or github_url, not both")

        run_id = str(uuid.uuid4())
        run_dir = self.run_artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        context = self.context_builder.build(task_prompt, mode=mode, cwd=cwd)
        config_path = context.write_config(run_dir / "cl_swe_agent_overlay.yaml")
        problem_statement_path = run_dir / "problem_statement.md"
        problem_statement_path.write_text(task_prompt, encoding="utf-8")
        resolved_output_dir = str(output_dir) if output_dir is not None else None

        args = self.build_command(
            config_path=str(config_path),
            problem_statement_path=str(problem_statement_path),
            task_id=task_id,
            repo_path=repo_path,
            github_url=github_url,
            model=model,
            output_dir=resolved_output_dir,
            extra_args=extra_args,
        )
        return SWEAgentCommandPreview(
            run_id=run_id,
            context=context,
            args=args,
            config_path=str(config_path),
            problem_statement_path=str(problem_statement_path),
            output_dir=resolved_output_dir,
        )

    def run(
        self,
        task_prompt: str,
        *,
        task_id: str,
        mode: Literal["baseline", "integrated"] = "baseline",
        cwd: str | None = None,
        repo_path: str | None = None,
        github_url: str | None = None,
        model: str | None = None,
        output_dir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] = (),
        timeout_seconds: int | None = None,
    ) -> SWEAgentRunResult:
        preview = self.preview(
            task_prompt,
            task_id=task_id,
            mode=mode,
            cwd=cwd,
            repo_path=repo_path,
            github_url=github_url,
            model=model,
            output_dir=output_dir,
            extra_args=extra_args,
        )
        effective_timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        process = self.command_runner(preview.args, cwd, env, effective_timeout)
        return SWEAgentRunResult(preview=preview, process=process)

    @staticmethod
    def _subprocess_runner(
        args: Sequence[str],
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_seconds: int | None = None,
    ) -> SWEAgentProcessResult:
        try:
            completed = subprocess.run(
                list(args),
                cwd=cwd,
                env={**os.environ, **dict(env or {})},
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raw_stdout = getattr(exc, "stdout", None)
            if raw_stdout is None:
                raw_stdout = exc.output
            raw_stderr = exc.stderr
            stdout = raw_stdout.decode("utf-8", errors="replace") if isinstance(raw_stdout, bytes) else raw_stdout
            stderr = raw_stderr.decode("utf-8", errors="replace") if isinstance(raw_stderr, bytes) else raw_stderr
            timeout_message = f"SWE-agent command timed out after {timeout_seconds} seconds."
            return SWEAgentProcessResult(
                args=list(args),
                returncode=124,
                stdout=stdout or "",
                stderr=f"{stderr or ''}\n{timeout_message}".strip(),
            )
        return SWEAgentProcessResult(
            args=list(args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    @staticmethod
    def _unconfigured_runner(
        _args: Sequence[str],
        _cwd: str | None,
        _env: Mapping[str, str] | None,
        _timeout_seconds: int | None = None,
    ) -> SWEAgentProcessResult:
        raise RuntimeError(
            "SWEAgentRunner requires an injected command_runner for execution. "
            "Use preview() for command/config generation or import saved .traj files."
        )
