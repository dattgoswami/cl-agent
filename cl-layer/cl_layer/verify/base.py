"""Base types for the verifier framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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


# --------------- Runner protocol ---------------


class VerificationRunner(Protocol):
    """Interface for a verification runner."""

    def run(self, repo_path: str, extra_env: dict[str, str] | None = None) -> VerificationResult: ...
