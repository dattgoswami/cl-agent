"""Tests for promotion gates."""

from __future__ import annotations

import pytest

from cl_layer.train.promotion import PromotionGate, PromotionResult
from cl_layer.eval.modes import EvalMode
from cl_layer.train.promotion import EvaluationResult


def _make_eval(mode: EvalMode = "search_sft", completion_rate: float = 0.8, latency_ms: float = 5000) -> EvaluationResult:
    return EvaluationResult(
        mode=mode,
        holdout_name="test_holdout",
        completion_rate=completion_rate,
        regression_rate=0.01,
        latency_ms=latency_ms,
    )


class TestPromotionGate:
    def test_promote_when_all_pass(self):
        gate = PromotionGate(primary_lift_threshold=0.05)
        new = _make_eval(completion_rate=0.85)
        current = _make_eval(completion_rate=0.8)
        result = gate.evaluate(new, current)
        assert result.decision == "promote"
        assert result.reason == "All gates passed"
        assert result.primary_lift > 0

    def test_reject_when_no_lift(self):
        gate = PromotionGate(primary_lift_threshold=0.05)
        new = _make_eval(completion_rate=0.8)
        current = _make_eval(completion_rate=0.8)
        result = gate.evaluate(new, current)
        assert result.decision == "reject"

    def test_reject_when_below_threshold(self):
        gate = PromotionGate(primary_lift_threshold=0.10)  # 10% threshold
        new = _make_eval(completion_rate=0.85)
        current = _make_eval(completion_rate=0.8)  # only 6.25% lift
        result = gate.evaluate(new, current)
        assert result.decision == "reject"
        assert "lift" in result.reason.lower()

    def test_reject_on_smoke_failure(self):
        gate = PromotionGate()
        new = _make_eval(completion_rate=0.9)
        current = _make_eval(completion_rate=0.8)
        result = gate.evaluate(new, current, smoke_passed=False)
        assert result.decision == "reject"
        assert "smoke" in result.reason.lower()
        assert result.smoke_passed is False

    def test_reject_on_regression(self):
        gate = PromotionGate(legacy_regression_threshold=0.02)
        new = EvaluationResult(
            mode="search_sft",
            holdout_name="test",
            completion_rate=0.85,
            regression_rate=0.05,  # high regression
            latency_ms=5000,
        )
        current = EvaluationResult(
            mode="search_sft",
            holdout_name="test",
            completion_rate=0.80,
            regression_rate=0.01,
            latency_ms=5000,
        )
        result = gate.evaluate(new, current)
        assert result.decision == "reject"
        assert "regression" in result.reason.lower()

    def test_reject_on_latency(self):
        gate = PromotionGate(latency_bound_ms=30_000)
        new = EvaluationResult(
            mode="search_sft",
            holdout_name="test",
            completion_rate=0.85,
            latency_ms=50_000,  # too slow
        )
        current = _make_eval(completion_rate=0.80, latency_ms=5000)
        result = gate.evaluate(new, current)
        assert result.decision == "reject"
        assert "latency" in result.reason.lower()
        assert result.latency_ok is False

    def test_should_rollback(self):
        gate = PromotionGate()
        current = EvaluationResult(mode="search_sft", holdout_name="current", completion_rate=0.50)
        fallback = EvaluationResult(mode="search_sft", holdout_name="fallback", completion_rate=0.80)
        assert gate.should_rollback(current, fallback) is True

    def test_should_not_rollback(self):
        gate = PromotionGate()
        current = EvaluationResult(mode="search_sft", holdout_name="current", completion_rate=0.80)
        fallback = EvaluationResult(mode="search_sft", holdout_name="fallback", completion_rate=0.50)
        assert gate.should_rollback(current, fallback) is False
