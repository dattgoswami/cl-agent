"""Base types for the verifier framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class VerificationStep:
    """Result of a single verification step."""

    name: str
    command: list[str]
    cwd: str
    exit_code: int
    duration_ms: float
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    success: bool = False


@dataclass
class VerificationResult:
    """Aggregated result of a full verification run."""

    task_id: str
    success: bool
    score: float
    steps: list[VerificationStep]
    failures: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)


@dataclass
class CommandResult:
    """Outcome of running a single command via a :class:`CommandRunner`.

    ``stdout`` and ``stderr`` are bytes so the verifier owns decoding policy
    (with ``errors="replace"``) the same way it would for ``subprocess.run``.
    """

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class CommandRunner(Protocol):
    """Injectable command runner. Default is :class:`SubprocessRunner`.

    Implementations MUST raise :class:`subprocess.TimeoutExpired` on timeout
    so the verifier can record a timeout failure consistently. They MUST
    NOT use ``shell=True``.
    """

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult: ...


# --------------- Verification runner protocol ---------------


class VerificationRunner(Protocol):
    """Interface for a verification runner."""

    def run(
        self,
        repo_path: str,
        extra_env: dict[str, str] | None = None,
        task_id: str = "",
    ) -> VerificationResult: ...
