"""Generic Python repo verifier that runs a sequence of steps.

Execution is injectable via a :class:`CommandRunner`. The default
:class:`SubprocessRunner` wraps ``subprocess.run`` with ``shell=False`` and
preserves per-step timeouts. Tests can pass a fake runner instead of
monkey-patching ``subprocess.run`` at module scope.

After running the configured steps, the verifier best-effort populates
``changed_files`` from ``git status --porcelain`` in the repo path. If the
path is not a git repo or the git command fails, ``changed_files`` is
``[]`` — extraction failures never mask verifier outcomes.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .base import (
    CommandResult,
    CommandRunner,
    VerificationResult,
    VerificationStep,
)


class SubprocessRunner:
    """Default :class:`CommandRunner` backed by ``subprocess.run``."""

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        result = subprocess.run(
            command,
            shell=False,
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout or b"",
            stderr=result.stderr or b"",
        )


def extract_changed_files(
    repo_path: str,
    runner: CommandRunner | None = None,
    timeout: float = 10.0,
) -> list[str]:
    """Best-effort: list of files changed in ``repo_path`` via git porcelain.

    Returns ``[]`` if the path is not a git repo, the runner raises, or
    the command exits non-zero. Never raises.
    """
    runner = runner or SubprocessRunner()

    if not (Path(repo_path) / ".git").exists():
        return []
    try:
        result = runner.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    files: list[str] = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        # porcelain format: "XY path" or "R  old -> new" for renames.
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if path:
            files.append(path)
    return files


class PythonRepoVerifier:
    """Run a sequence of verification commands against a Python repo."""

    def __init__(
        self,
        steps: list[dict],
        runner: CommandRunner | None = None,
    ) -> None:
        # Each step: {"name", "command", "cwd"?, "expected_exit_code"?, "timeout"?}
        self.steps = steps
        self._runner: CommandRunner = runner or SubprocessRunner()

    def run(
        self,
        repo_path: str,
        extra_env: dict[str, str] | None = None,
        task_id: str = "",
    ) -> VerificationResult:
        steps: list[VerificationStep] = []
        failures: list[str] = []
        success = True

        env = {**os.environ, **(extra_env or {})}

        for step_def in self.steps:
            name = step_def["name"]
            cmd = step_def["command"]
            cwd = step_def.get("cwd", repo_path)
            expected_exit = step_def.get("expected_exit_code", 0)
            timeout = step_def.get("timeout", 120)

            start = time.monotonic()
            try:
                result = self._runner.run(
                    cmd,
                    cwd=cwd,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                elapsed = (time.monotonic() - start) * 1000
                steps.append(
                    VerificationStep(
                        name=name,
                        command=cmd,
                        cwd=cwd,
                        exit_code=-1,
                        duration_ms=elapsed,
                        stdout_excerpt="",
                        stderr_excerpt="timeout",
                        success=False,
                    )
                )
                failures.append(f"{name}: timeout")
                success = False
                continue

            elapsed = (time.monotonic() - start) * 1000
            step_success = result.returncode == expected_exit
            steps.append(
                VerificationStep(
                    name=name,
                    command=cmd,
                    cwd=cwd,
                    exit_code=result.returncode,
                    duration_ms=elapsed,
                    stdout_excerpt=result.stdout[:500].decode("utf-8", errors="replace"),
                    stderr_excerpt=result.stderr[:500].decode("utf-8", errors="replace"),
                    success=step_success,
                )
            )
            if not step_success:
                success = False
                failures.append(
                    f"{name}: expected exit {expected_exit}, got {result.returncode}"
                )

        score = _compute_score(steps)
        changed_files = extract_changed_files(repo_path, self._runner)

        return VerificationResult(
            task_id=task_id,
            success=success,
            score=score,
            steps=steps,
            failures=failures,
            changed_files=changed_files,
        )

    @staticmethod
    def from_steps(
        steps: list[dict], runner: CommandRunner | None = None
    ) -> "PythonRepoVerifier":
        return PythonRepoVerifier(steps, runner=runner)


def _compute_score(steps: list[VerificationStep]) -> float:
    if not steps:
        return 0.0
    return sum(1.0 for s in steps if s.success) / len(steps)
