"""Mode-comparison runner for `baseline` / `symbolic` / `search` / `search_sft`.

Each mode is represented by an injected callable: ``Callable[[BenchmarkTask],
TaskAttempt]``. The runner is deliberately ignorant of how each mode
actually solves a task — that lets unit tests pass deterministic fakes
without touching real models, Ollama, search loops, or the network.

The aggregated :class:`ModeReport` exposes ``completion_rate``,
``regression_rate``, ``mean_latency_ms``, and ``mean_edit_size``, and can
be converted to an :class:`EvaluationResult` to feed
:class:`PromotionGate` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from cl_layer.eval.benchmark import BenchmarkSuite, BenchmarkTask
from cl_layer.eval.modes import KNOWN_MODES, EvalMode
from cl_layer.train.promotion import EvaluationResult


@dataclass
class TaskAttempt:
    """One mode's attempt at one benchmark task."""

    task_id: str
    success: bool
    latency_ms: float
    regressions: int = 0
    edit_size: int = 0
    metadata: dict = field(default_factory=dict)


ModeSolver = Callable[[BenchmarkTask], TaskAttempt]


@dataclass
class ModeReport:
    """Aggregate of one mode's attempts across a benchmark slice."""

    mode: EvalMode
    holdout_name: str
    attempts: list[TaskAttempt] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.attempts)

    @property
    def completion_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(1 for a in self.attempts if a.success) / len(self.attempts)

    @property
    def regression_rate(self) -> float:
        """Mean number of regressions per task. Zero when no attempts."""
        if not self.attempts:
            return 0.0
        return sum(a.regressions for a in self.attempts) / len(self.attempts)

    @property
    def mean_latency_ms(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(a.latency_ms for a in self.attempts) / len(self.attempts)

    @property
    def mean_edit_size(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(a.edit_size for a in self.attempts) / len(self.attempts)

    def to_evaluation_result(self) -> EvaluationResult:
        """Convert this report into the surface :class:`PromotionGate` accepts."""
        return EvaluationResult(
            mode=self.mode,
            holdout_name=self.holdout_name,
            completion_rate=self.completion_rate,
            regression_rate=self.regression_rate,
            mean_edit_size=self.mean_edit_size,
            latency_ms=self.mean_latency_ms,
            pass_rate=self.completion_rate,
            test_scores={
                "n": self.n,
                "completed": sum(1 for a in self.attempts if a.success),
            },
        )


def run_modes(
    suite: BenchmarkSuite,
    solvers: dict[EvalMode, ModeSolver],
    *,
    holdout_name: str = "holdout",
    split: str | None = None,
) -> dict[EvalMode, ModeReport]:
    """Run each provided mode solver against the suite (or a split of it).

    Unknown mode keys raise :class:`ValueError`. Missing modes are simply
    not reported. Each solver runs on every task in (the filtered) suite.
    """
    for mode in solvers:
        if mode not in KNOWN_MODES:
            raise ValueError(
                f"Unknown evaluation mode: {mode!r}. Expected one of {KNOWN_MODES}"
            )

    tasks = suite.filter_split(split)
    reports: dict[EvalMode, ModeReport] = {}
    for mode, solver in solvers.items():
        attempts: list[TaskAttempt] = []
        for task in tasks:
            attempt = solver(task)
            attempts.append(attempt)
        reports[mode] = ModeReport(
            mode=mode, holdout_name=holdout_name, attempts=attempts
        )
    return reports


def compare_modes(
    reports: dict[EvalMode, ModeReport],
    new_mode: EvalMode,
    current_mode: EvalMode,
) -> tuple[EvaluationResult, EvaluationResult]:
    """Pull two :class:`EvaluationResult` records out of a report bundle.

    Useful when feeding :class:`PromotionGate.evaluate` to compare a
    candidate mode (e.g., ``search_sft``) against the currently promoted
    one (e.g., ``symbolic``).
    """
    if new_mode not in reports:
        raise KeyError(f"Mode {new_mode!r} not in reports")
    if current_mode not in reports:
        raise KeyError(f"Mode {current_mode!r} not in reports")
    return (
        reports[new_mode].to_evaluation_result(),
        reports[current_mode].to_evaluation_result(),
    )
