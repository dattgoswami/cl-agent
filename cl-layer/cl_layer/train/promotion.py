"""Promotion gates for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cl_layer.eval.modes import EvalMode

PromotionDecision = Literal["promote", "reject", "rollback"]


@dataclass
class EvaluationResult:
    """Result of evaluating a model on a holdout set."""

    mode: EvalMode
    holdout_name: str
    completion_rate: float
    regression_rate: float = 0.0
    mean_edit_size: float = 0.0
    latency_ms: float = 0.0
    pass_rate: float = 0.0
    test_scores: dict = field(default_factory=dict)


@dataclass
class PromotionResult:
    """Result of a promotion gate evaluation."""

    decision: PromotionDecision
    reason: str
    primary_lift: float = 0.0
    legacy_regression: float = 0.0
    smoke_passed: bool = True
    latency_ok: bool = True


class PromotionGate:
    """Gate that decides whether to promote a new model.

    Rules:
    - minimum 5% relative lift on primary holdout
    - maximum 2% absolute regression on legacy holdout
    - smoke prompts pass
    - latency within bound
    """

    def __init__(
        self,
        primary_lift_threshold: float = 0.05,
        legacy_regression_threshold: float = 0.02,
        latency_bound_ms: float = 30_000,
    ) -> None:
        self.primary_lift_threshold = primary_lift_threshold
        self.legacy_regression_threshold = legacy_regression_threshold
        self.latency_bound_ms = latency_bound_ms

    def evaluate(
        self,
        new_model: EvaluationResult,
        current_model: EvaluationResult,
        smoke_passed: bool = True,
    ) -> PromotionResult:
        # Check smoke
        if not smoke_passed:
            return PromotionResult(
                decision="reject",
                reason="Smoke prompts failed",
                smoke_passed=False,
            )

        # Check primary lift
        primary_lift = (new_model.completion_rate - current_model.completion_rate) / max(current_model.completion_rate, 1e-9)
        if primary_lift < self.primary_lift_threshold:
            return PromotionResult(
                decision="reject",
                reason=f"Primary lift {primary_lift:.4f} below threshold {self.primary_lift_threshold}",
                primary_lift=primary_lift,
            )

        # Check legacy regression
        legacy_regression = new_model.regression_rate - current_model.regression_rate
        if legacy_regression > self.legacy_regression_threshold:
            return PromotionResult(
                decision="reject",
                reason=f"Legacy regression {legacy_regression:.4f} exceeds threshold {self.legacy_regression_threshold}",
                legacy_regression=legacy_regression,
            )

        # Check latency
        latency_ok = new_model.latency_ms <= self.latency_bound_ms
        if not latency_ok:
            return PromotionResult(
                decision="reject",
                reason=f"Latency {new_model.latency_ms:.0f}ms exceeds bound {self.latency_bound_ms}ms",
                latency_ok=False,
            )

        return PromotionResult(
            decision="promote",
            reason="All gates passed",
            primary_lift=primary_lift,
            legacy_regression=legacy_regression,
            smoke_passed=smoke_passed,
            latency_ok=latency_ok,
        )

    def should_rollback(self, current: EvaluationResult, fallback: EvaluationResult) -> bool:
        """Check if the current model should be rolled back."""
        return current.completion_rate < fallback.completion_rate * 0.95
