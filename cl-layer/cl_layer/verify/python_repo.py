"""Generic Python repo verifier that runs a sequence of steps."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .base import VerificationResult, VerificationStep


class PythonRepoVerifier:
    """Run a sequence of verification commands against a Python repo."""

    def __init__(self, steps: list[dict]) -> None:
        # Each step: {"name", "command", "cwd", "expected_exit_code"}
        self.steps = steps

    def run(
        self,
        repo_path: str,
        extra_env: dict[str, str] | None = None,
    ) -> VerificationResult:
        steps: list[VerificationStep] = []
        failures: list[str] = []
        success = True
        changed_files: list[str] = []

        for step_def in self.steps:
            name = step_def["name"]
            cmd = step_def["command"]
            cwd = step_def.get("cwd", repo_path)
            expected_exit = step_def.get("expected_exit_code", 0)

            start = time.monotonic()
            try:
                result = subprocess.run(
                    cmd,
                    shell=False,
                    cwd=cwd,
                    capture_output=True,
                    timeout=step_def.get("timeout", 120),
                    env={**__import__("os").environ, **(extra_env or {})},
                )
            except subprocess.TimeoutExpired:
                elapsed = (time.monotonic() - start) * 1000
                step = VerificationStep(
                    name=name,
                    command=cmd,
                    cwd=cwd,
                    exit_code=-1,
                    duration_ms=elapsed,
                    stdout_excerpt="",
                    stderr_excerpt="timeout",
                    success=False,
                )
                steps.append(step)
                failures.append(f"{name}: timeout")
                success = False
                continue

            elapsed = (time.monotonic() - start) * 1000
            step_success = result.returncode == expected_exit
            step = VerificationStep(
                name=name,
                command=cmd,
                cwd=cwd,
                exit_code=result.returncode,
                duration_ms=elapsed,
                stdout_excerpt=(result.stdout or b"")[:500].decode("utf-8", errors="replace"),
                stderr_excerpt=(result.stderr or b"")[:500].decode("utf-8", errors="replace"),
                success=step_success,
            )
            steps.append(step)
            if not step_success:
                success = False
                failures.append(f"{name}: expected exit {expected_exit}, got {result.returncode}")

        score = _compute_score(steps)
        return VerificationResult(
            task_id="",
            success=success,
            score=score,
            steps=steps,
            failures=failures,
            changed_files=changed_files,
        )

    @staticmethod
    def from_steps(steps: list[dict]) -> PythonRepoVerifier:
        return PythonRepoVerifier(steps)


def _compute_score(steps: list[VerificationStep]) -> float:
    if not steps:
        return 0.0
    return sum(1.0 for s in steps if s.success) / len(steps)
