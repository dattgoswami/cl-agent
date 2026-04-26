"""Tests for verifier framework and scoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cl_layer.verify.base import VerificationResult, VerificationStep
from cl_layer.verify.python_repo import PythonRepoVerifier, _compute_score
from cl_layer.verify.score import (
    score_build_status,
    score_lint_status,
    score_novelty_bonus,
    score_patch_size,
    score_regressions,
    score_runtime_cost,
    score_tests_fixed,
    score_type_status,
)


class TestVerificationStep:
    def test_defaults(self):
        step = VerificationStep(
            name="test",
            command=["pytest"],
            cwd="/tmp",
            exit_code=0,
            duration_ms=100.0,
        )
        assert step.stdout_excerpt == ""
        assert step.stderr_excerpt == ""
        assert step.success is False

    def test_success_flag(self):
        step = VerificationStep(
            name="test",
            command=["pytest"],
            cwd="/tmp",
            exit_code=0,
            duration_ms=100.0,
            success=True,
        )
        assert step.success is True


class TestPythonRepoVerifier:
    def test_run_all_steps_succeed(self):
        steps = [
            {"name": "lint", "command": ["ruff", "check"], "expected_exit_code": 0, "timeout": 10},
            {"name": "test", "command": ["python", "-m", "pytest"], "expected_exit_code": 0, "timeout": 10},
        ]
        verifier = PythonRepoVerifier(steps)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"ok\n"
        mock_result.stderr = b""
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = verifier.run("/tmp/repo")
        assert result.success is True
        assert result.score == pytest.approx(1.0)
        assert len(result.steps) == 2
        assert result.failures == []

    def test_run_step_fails(self):
        steps = [
            {"name": "lint", "command": ["ruff", "check"], "expected_exit_code": 0, "timeout": 10},
        ]
        verifier = PythonRepoVerifier(steps)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"error\n"
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = verifier.run("/tmp/repo")
        assert result.success is False
        assert result.score == pytest.approx(0.0)
        assert len(result.failures) == 1

    def test_run_with_timeout(self):
        steps = [
            {"name": "test", "command": ["python", "-m", "pytest"], "expected_exit_code": 0, "timeout": 1},
        ]
        verifier = PythonRepoVerifier(steps)
        import subprocess
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python"], timeout=1)
            result = verifier.run("/tmp/repo")
        assert result.success is False
        assert any("timeout" in f for f in result.failures)

    def test_empty_steps(self):
        verifier = PythonRepoVerifier([])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""
        mock_result.stderr = b""
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = verifier.run("/tmp/repo")
        assert result.score == pytest.approx(0.0)


class TestComputeScore:
    def test_all_pass(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert _compute_score(steps) == pytest.approx(1.0)

    def test_half_fail(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
        ]
        assert _compute_score(steps) == pytest.approx(0.5)


# --------------- scoring helpers ------------

class TestScoringHelpers:
    def test_score_tests_fixed(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
            VerificationStep(name="c", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_tests_fixed(steps) == 2

    def test_score_regressions(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
        ]
        assert score_regressions(steps) == 1

    def test_score_lint_status_all_pass(self):
        steps = [
            VerificationStep(name="lint_ruff", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_lint_status(steps) == 1.0

    def test_score_lint_status_fails(self):
        steps = [
            VerificationStep(name="lint_ruff", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
        ]
        assert score_lint_status(steps) == 0.0

    def test_score_lint_status_no_lint_steps(self):
        assert score_lint_status([]) == 1.0

    def test_score_type_status_all_pass(self):
        steps = [
            VerificationStep(name="typecheck_mypy", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_type_status(steps) == 1.0

    def test_score_type_status_no_steps(self):
        assert score_type_status([]) == 1.0

    def test_score_build_status_all_pass(self):
        steps = [
            VerificationStep(name="build", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_build_status(steps) == 1.0

    def test_score_build_status_no_steps(self):
        assert score_build_status([]) == 1.0

    def test_score_patch_size_small(self):
        assert score_patch_size("line1\nline2\n") == 1.0

    def test_score_patch_size_giant(self):
        big = "\n".join(f"line{i}" for i in range(300))
        assert score_patch_size(big) < 1.0

    def test_score_patch_size_none(self):
        assert score_patch_size(None) == 1.0

    def test_score_runtime_cost_under_max(self):
        assert score_runtime_cost(10_000) == 1.0

    def test_score_runtime_cost_over_max(self):
        assert score_runtime_cost(200_000) < 1.0

    def test_score_runtime_cost_zero(self):
        assert score_runtime_cost(0) == 1.0

    def test_score_novelty_bonus_all_new(self):
        assert score_novelty_bonus(["a.py", "b.py"], {"c.py"}) == 1.0

    def test_score_novelty_bonus_some_new(self):
        assert score_novelty_bonus(["a.py", "b.py"], {"a.py"}) == pytest.approx(0.5)

    def test_score_novelty_bonus_all_known(self):
        assert score_novelty_bonus(["a.py"], {"a.py"}) == 0.0

    def test_score_novelty_bonus_no_archive(self):
        assert score_novelty_bonus(["a.py", "b.py"]) == 1.0
