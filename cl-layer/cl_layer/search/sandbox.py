"""Sandbox abstraction for applying candidate patches.

The SOAR controller MUST NOT mutate the user's live repo. Every candidate
is applied through a ``Sandbox`` that produces an isolated workspace path;
the verifier then runs against that path. Production implementations
should snapshot a base repo into a temp directory and apply the patch
there. ``InMemorySandbox`` is a research/test stub that fabricates paths
without touching the filesystem and is suitable for unit tests that mock
the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cl_layer.search.base import Candidate


@dataclass
class AppliedCandidate:
    """Result of applying a candidate to a sandbox."""

    candidate: Candidate
    sandbox_path: str
    changed_files: list[str] = field(default_factory=list)


class Sandbox(Protocol):
    """Apply a candidate's patch to an isolated workspace.

    The verifier receives only ``AppliedCandidate.sandbox_path``; raw patch
    text is never passed across the boundary.
    """

    def apply(
        self, candidate: Candidate, *, task_id: str | None = None
    ) -> AppliedCandidate: ...

    def cleanup(self, applied: AppliedCandidate) -> None: ...


class InMemorySandbox:
    """Fabricates per-candidate paths without writing to disk.

    Suitable for unit tests where the verifier is also mocked. Records
    every ``apply`` and ``cleanup`` for assertion. NOT for production.
    """

    def __init__(self, base_path: str = "/tmp/cl-agent-sandbox") -> None:
        self.base_path = base_path
        self.applied: list[Candidate] = []
        self.cleaned: list[str] = []

    def apply(
        self, candidate: Candidate, *, task_id: str | None = None
    ) -> AppliedCandidate:
        self.applied.append(candidate)
        sub = task_id or "default"
        path = f"{self.base_path}/{sub}/{candidate.id}"
        return AppliedCandidate(
            candidate=candidate,
            sandbox_path=path,
            changed_files=list(candidate.affected_files),
        )

    def cleanup(self, applied: AppliedCandidate) -> None:
        self.cleaned.append(applied.sandbox_path)
