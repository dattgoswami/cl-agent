from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import Episode, new_episode_id

from .context_builder import AiderRunContext, ContextBuilder
from .item_mapper import map_aider_run
from .log_loader import load_chat_history


@dataclass(frozen=True)
class AiderProcessResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GitSnapshot:
    is_repo: bool
    head: str | None
    status_text: str
    dirty_files: list[str]
    diff_text: str | None
    error: str | None = None


@dataclass(frozen=True)
class AiderCommandResult:
    context: AiderRunContext
    process: AiderProcessResult
    episode: Episode
    chat_history_file: str
    input_history_file: str
    llm_history_file: str


CommandRunner = Callable[[Sequence[str], str | None, Mapping[str, str] | None], AiderProcessResult]
GitRunner = Callable[[Sequence[str], str | None], AiderProcessResult]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def parse_git_status_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for raw in status_text.splitlines():
        if not raw:
            continue
        path = raw[3:] if len(raw) > 3 else raw
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip()
        if path:
            paths.append(path)
    return sorted(dict.fromkeys(paths))


class AiderRunner:
    """
    Thin, injectable Aider CLI runner.

    Capture is built from subprocess output, run-specific Aider history files,
    and git snapshots before/after the run.  Tests can inject command runners
    so no Aider binary, model key, package install, or network is required.
    """

    def __init__(
        self,
        episodes_path: str | Path,
        *,
        artifacts_dir: str | Path | None = None,
        capture_dir: str | Path | None = None,
        aider_executable: str = "aider",
        command_runner: CommandRunner | None = None,
        git_runner: GitRunner | None = None,
    ) -> None:
        self.episodes_path = Path(episodes_path)
        self.recorder = EpisodeRecorder(self.episodes_path)
        self.context_builder = ContextBuilder(artifacts_dir)
        self.capture_dir = Path(capture_dir) if capture_dir else self.episodes_path.parent / "aider-captures"
        self.aider_executable = aider_executable
        self.command_runner = command_runner or self._subprocess_runner
        self.git_runner = git_runner or self._git_runner

    def build_command(
        self,
        context: AiderRunContext,
        *,
        chat_history_file: str,
        input_history_file: str,
        llm_history_file: str,
        model: str | None = None,
        files: Sequence[str] = (),
        lint_cmds: Sequence[str] = (),
        test_cmd: str | None = None,
        auto_test: bool = False,
        extra_args: Sequence[str] = (),
    ) -> list[str]:
        args = [
            self.aider_executable,
            "--no-auto-commits",
            "--no-dirty-commits",
            "--no-restore-chat-history",
            "--no-gitignore",
            "--no-analytics",
            "--no-check-update",
            "--no-show-release-notes",
            "--yes-always",
            "--no-pretty",
            "--no-stream",
            "--chat-history-file",
            chat_history_file,
            "--input-history-file",
            input_history_file,
            "--llm-history-file",
            llm_history_file,
        ]
        if model:
            args.extend(["--model", model])
        for lint_cmd in lint_cmds:
            args.extend(["--lint-cmd", lint_cmd])
        if test_cmd:
            args.extend(["--test-cmd", test_cmd])
        if auto_test:
            args.append("--auto-test")
        args.extend(extra_args)
        args.extend(files)
        args.extend(context.cli_args())
        return args

    def run(
        self,
        task_prompt: str,
        *,
        task_id: str,
        task_domain: str,
        mode: Literal["baseline", "integrated"] = "baseline",
        cwd: str | None = None,
        task_description: str | None = None,
        model: str | None = None,
        files: Sequence[str] = (),
        lint_cmds: Sequence[str] = (),
        test_cmd: str | None = None,
        auto_test: bool = False,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] = (),
        agent_surface: str = "aider",
    ) -> Episode:
        result = self.run_with_result(
            task_prompt,
            task_id=task_id,
            task_domain=task_domain,
            mode=mode,
            cwd=cwd,
            task_description=task_description,
            model=model,
            files=files,
            lint_cmds=lint_cmds,
            test_cmd=test_cmd,
            auto_test=auto_test,
            env=env,
            extra_args=extra_args,
            agent_surface=agent_surface,
        )
        return result.episode

    def run_with_result(
        self,
        task_prompt: str,
        *,
        task_id: str,
        task_domain: str,
        mode: Literal["baseline", "integrated"] = "baseline",
        cwd: str | None = None,
        task_description: str | None = None,
        model: str | None = None,
        files: Sequence[str] = (),
        lint_cmds: Sequence[str] = (),
        test_cmd: str | None = None,
        auto_test: bool = False,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] = (),
        agent_surface: str = "aider",
    ) -> AiderCommandResult:
        if mode not in {"baseline", "integrated"}:
            raise ValueError("mode must be 'baseline' or 'integrated'")

        run_id = str(uuid.uuid4())
        run_capture_dir = self.capture_dir / run_id
        run_capture_dir.mkdir(parents=True, exist_ok=True)
        chat_history_file = str(run_capture_dir / "chat-history.md")
        input_history_file = str(run_capture_dir / "input-history")
        llm_history_file = str(run_capture_dir / "llm-history.txt")

        context = self.context_builder.build(task_prompt, mode=mode, cwd=cwd)
        args = self.build_command(
            context,
            chat_history_file=chat_history_file,
            input_history_file=input_history_file,
            llm_history_file=llm_history_file,
            model=model,
            files=files,
            lint_cmds=lint_cmds,
            test_cmd=test_cmd,
            auto_test=auto_test,
            extra_args=extra_args,
        )

        before = self._snapshot_git(cwd)
        started_at = _now()
        process = self.command_runner(args, cwd, env)
        ended_at = _now()
        after = self._snapshot_git(cwd)
        patch_text, changed_files = self._derive_patch_and_files(before, after, cwd)
        chat_messages = load_chat_history(chat_history_file)

        mapping = map_aider_run(
            args=process.args,
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            cwd=cwd,
            started_at=started_at,
            ended_at=ended_at,
            chat_messages=chat_messages,
            changed_files=changed_files,
            patch_text=patch_text,
            preexisting_dirty_files=before.dirty_files,
            commit_before=before.head,
            commit_after=after.head,
        )

        episode = Episode(
            episode_id=new_episode_id(),
            run_id=run_id,
            thread_id=None,
            task_id=task_id,
            task_description=task_description if task_description is not None else task_prompt[:200],
            task_domain=task_domain,
            agent_surface=agent_surface,
            mode=mode,
            started_at=started_at,
            ended_at=ended_at,
            events=mapping.events,
            outcome=mapping.outcome,
            reward=None,
            repo_path=cwd,
            git_commit=after.head,
            base_model_id=model,
            patch_text=mapping.patch_text,
            patch_hash=mapping.patch_hash,
            tool_trace=mapping.tool_trace,
            test_trace=mapping.test_trace,
            stdout_excerpt=mapping.stdout_excerpt,
            stderr_excerpt=mapping.stderr_excerpt,
            latency_ms=mapping.latency_ms,
        )

        self.recorder.append(episode)
        return AiderCommandResult(
            context=context,
            process=process,
            episode=episode,
            chat_history_file=chat_history_file,
            input_history_file=input_history_file,
            llm_history_file=llm_history_file,
        )

    def _snapshot_git(self, cwd: str | None) -> GitSnapshot:
        root = self.git_runner(["git", "rev-parse", "--show-toplevel"], cwd)
        if root.returncode != 0:
            return GitSnapshot(
                is_repo=False,
                head=None,
                status_text="",
                dirty_files=[],
                diff_text=None,
                error=(root.stderr or root.stdout or "not a git repository"),
            )

        head_res = self.git_runner(["git", "rev-parse", "HEAD"], cwd)
        head = head_res.stdout.strip() if head_res.returncode == 0 and head_res.stdout.strip() else None

        status_res = self.git_runner(["git", "status", "--porcelain=v1"], cwd)
        status_text = status_res.stdout if status_res.returncode == 0 else ""
        dirty_files = parse_git_status_paths(status_text)

        diff_args = ["git", "diff", "--no-ext-diff", "--binary"]
        if head:
            diff_args.append("HEAD")
        diff_args.append("--")
        diff_res = self.git_runner(diff_args, cwd)
        diff_text = diff_res.stdout if diff_res.returncode == 0 and diff_res.stdout else None

        return GitSnapshot(
            is_repo=True,
            head=head,
            status_text=status_text,
            dirty_files=dirty_files,
            diff_text=diff_text,
            error=None,
        )

    def _derive_patch_and_files(
        self,
        before: GitSnapshot,
        after: GitSnapshot,
        cwd: str | None,
    ) -> tuple[str | None, list[str]]:
        if not before.is_repo or not after.is_repo:
            return None, []

        if before.head and after.head and before.head != after.head:
            diff_res = self.git_runner(["git", "diff", "--no-ext-diff", "--binary", before.head, after.head, "--"], cwd)
            name_res = self.git_runner(["git", "diff", "--name-only", before.head, after.head, "--"], cwd)
            patch = diff_res.stdout if diff_res.returncode == 0 and diff_res.stdout else None
            paths = name_res.stdout.splitlines() if name_res.returncode == 0 else []
            return patch, sorted(dict.fromkeys(path.strip() for path in paths if path.strip()))

        before_dirty = set(before.dirty_files)
        after_dirty = set(after.dirty_files)
        changed = sorted(after_dirty - before_dirty)

        if before.diff_text:
            # Avoid attributing a pre-existing dirty diff to Aider.  We keep the
            # new path delta, but skip patch_text unless the worktree was clean.
            return None, changed

        return after.diff_text, changed

    @staticmethod
    def _subprocess_runner(
        args: Sequence[str],
        cwd: str | None,
        env: Mapping[str, str] | None,
    ) -> AiderProcessResult:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env={**os.environ, **dict(env or {})},
            capture_output=True,
            text=True,
            check=False,
        )
        return AiderProcessResult(
            args=list(args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    @staticmethod
    def _git_runner(args: Sequence[str], cwd: str | None) -> AiderProcessResult:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return AiderProcessResult(
            args=list(args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
