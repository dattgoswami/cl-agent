"""Core dataclasses for the search package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A single search candidate."""

    id: str
    plan_text: str
    patch_text: str
    affected_files: list[str]
    verifier_score: float | None = None
    novelty_score: float = 0.0
    cost_score: float = 0.0
    parent_id: str | None = None
    generation: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Population:
    """A group of related candidates for a task."""

    population_id: str
    candidates: list[Candidate] = field(default_factory=list)
    task_id: str = ""
    generation: int = 0

    def add(self, candidate: Candidate) -> None:
        self.candidates.append(candidate)

    @property
    def size(self) -> int:
        return len(self.candidates)


@dataclass
class SearchConfig:
    """Configuration for the SOAR search loop."""

    k_candidates: int = 5
    n_elites: int = 3
    max_generations: int = 10
    max_candidates_total: int = 50
    novelty_window: int = 20
