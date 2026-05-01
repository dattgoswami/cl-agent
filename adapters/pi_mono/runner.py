from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from .context_builder import ContextBuilder, PiRunContext


@dataclass(frozen=True)
class PiCommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    context: PiRunContext


@dataclass(frozen=True)
class PiProcessResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], str | None, Mapping[str, str] | None], PiProcessResult]


class PiCliRunner:
    """
    Thin, injectable Pi CLI runner.

    This class intentionally does not parse live output into an Episode.  For
    capture, import the Pi session JSONL after the run.  Tests can inject a
    command runner so no Pi binary, API key, or model call is required.
    """

    def __init__(
        self,
        *,
        pi_executable: str = "pi",
        artifacts_dir: str | Path | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.pi_executable = pi_executable
        self.context_builder = ContextBuilder(artifacts_dir)
        self.command_runner = command_runner or self._subprocess_runner

    def build_command(
        self,
        context: PiRunContext,
        *,
        output_mode: str = "text",
        provider: str | None = None,
        model: str | None = None,
        session: str | None = None,
        extra_args: Sequence[str] = (),
    ) -> list[str]:
        args = [self.pi_executable]
        if output_mode == "json":
            args.extend(["--mode", "json"])
        else:
            args.append("--print")
        args.extend(context.cli_args())
        if provider:
            args.extend(["--provider", provider])
        if model:
            args.extend(["--model", model])
        if session:
            args.extend(["--session", session])
        args.extend(extra_args)
        args.append(context.task_prompt)
        return args

    def run(
        self,
        task_prompt: str,
        *,
        mode: Literal["baseline", "integrated"] = "baseline",
        cwd: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        session: str | None = None,
        output_mode: str = "text",
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] = (),
    ) -> PiCommandResult:
        if mode not in {"baseline", "integrated"}:
            raise ValueError("mode must be 'baseline' or 'integrated'")
        context = self.context_builder.build(task_prompt, mode=mode, cwd=cwd)
        args = self.build_command(
            context,
            output_mode=output_mode,
            provider=provider,
            model=model,
            session=session,
            extra_args=extra_args,
        )
        raw = self.command_runner(args, cwd, env)
        return PiCommandResult(
            args=raw.args,
            returncode=raw.returncode,
            stdout=raw.stdout,
            stderr=raw.stderr,
            context=context,
        )

    @staticmethod
    def _subprocess_runner(
        args: Sequence[str],
        cwd: str | None,
        env: Mapping[str, str] | None,
    ) -> PiProcessResult:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env={**os.environ, **dict(env or {})},
            capture_output=True,
            text=True,
            check=False,
        )
        return PiProcessResult(
            args=list(args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
