"""Tests for the eval package: benchmark loader + mode-comparison runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cl_layer.eval.benchmark import (
    KNOWN_CATEGORIES,
    BenchmarkLoadError,
    BenchmarkSuite,
    BenchmarkTask,
)
from cl_layer.eval.runner import (
    ModeReport,
    TaskAttempt,
    compare_modes,
    run_modes,
)
from cl_layer.train.promotion import EvaluationResult, PromotionGate


# --------------- fixtures ------------


def _suite_dict() -> dict:
    return {
        "name": "phase1-smoke",
        "tasks": [
            {
                "task_id": "repo-1",
                "prompt": "Add a /health endpoint.",
                "category": "repo_local",
                "split": "valid",
                "domain": "fastapi",
                "tags": ["endpoint"],
                "repo_path": "fixtures/repo",
                "verifier_commands": [
                    {"name": "pytest", "command": ["pytest"], "expected_exit_code": 0}
                ],
            },
            {
                "task_id": "ext-1",
                "prompt": "Fix off-by-one in ISO week numbers.",
                "category": "external_slice",
                "split": "test",
                "domain": "datetime",
            },
            {
                "task_id": "syn-1",
                "prompt": "Fix missing import.",
                "category": "synthetic_repair",
                "split": "train",
                "domain": "auth",
            },
        ],
    }


def _write_suite(tmp_path: Path, data: dict, name: str = "suite.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------- Loader ------------


class TestBenchmarkLoader:
    def test_loads_valid_json_suite(self, tmp_path):
        path = _write_suite(tmp_path, _suite_dict())
        suite = BenchmarkSuite.from_path(path)
        assert suite.name == "phase1-smoke"
        assert len(suite.tasks) == 3
        assert {t.category for t in suite.tasks} == {
            "repo_local",
            "external_slice",
            "synthetic_repair",
        }

    def test_loads_via_from_dict(self):
        suite = BenchmarkSuite.from_dict(_suite_dict())
        assert suite.tasks[0].verifier_commands[0]["name"] == "pytest"

    def test_loads_repo_root_fixture(self):
        # The repo's bundled fixture must load and have all three categories.
        fixture = Path(__file__).resolve().parents[2] / "benchmarks" / "example_suite.json"
        if not fixture.exists():
            pytest.skip(f"fixture missing: {fixture}")
        suite = BenchmarkSuite.from_path(fixture)
        assert {t.category for t in suite.tasks} == set(KNOWN_CATEGORIES)

    def test_split_filters(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        assert [t.task_id for t in suite.valid_tasks] == ["repo-1"]
        assert [t.task_id for t in suite.test_tasks] == ["ext-1"]
        assert [t.task_id for t in suite.train_tasks] == ["syn-1"]
        assert {t.task_id for t in suite.filter_split(None)} == {
            "repo-1",
            "ext-1",
            "syn-1",
        }

    def test_filter_category(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        repo_tasks = suite.filter_category("repo_local")
        assert len(repo_tasks) == 1
        assert repo_tasks[0].task_id == "repo-1"

    def test_default_category_is_repo_local(self):
        suite = BenchmarkSuite.from_dict(
            {
                "name": "x",
                "tasks": [{"task_id": "a", "prompt": "do x"}],
            }
        )
        assert suite.tasks[0].category == "repo_local"

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(BenchmarkLoadError, match="not found"):
            BenchmarkSuite.from_path(tmp_path / "nonexistent.json")

    def test_rejects_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(BenchmarkLoadError, match="Invalid JSON"):
            BenchmarkSuite.from_path(path)

    def test_rejects_non_dict_root(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(BenchmarkLoadError, match="must be a dict"):
            BenchmarkSuite.from_path(path)

    def test_rejects_missing_name(self, tmp_path):
        path = _write_suite(tmp_path, {"tasks": []})
        with pytest.raises(BenchmarkLoadError, match="'name'"):
            BenchmarkSuite.from_path(path)

    def test_rejects_missing_tasks(self, tmp_path):
        path = _write_suite(tmp_path, {"name": "x"})
        with pytest.raises(BenchmarkLoadError, match="'tasks'"):
            BenchmarkSuite.from_path(path)

    def test_rejects_tasks_not_a_list(self, tmp_path):
        path = _write_suite(tmp_path, {"name": "x", "tasks": {}})
        with pytest.raises(BenchmarkLoadError, match="must be a list"):
            BenchmarkSuite.from_path(path)

    def test_rejects_task_missing_required_fields(self, tmp_path):
        path = _write_suite(
            tmp_path, {"name": "x", "tasks": [{"task_id": "a"}]}  # no prompt
        )
        with pytest.raises(BenchmarkLoadError, match="'prompt'"):
            BenchmarkSuite.from_path(path)

    def test_rejects_invalid_category(self, tmp_path):
        path = _write_suite(
            tmp_path,
            {
                "name": "x",
                "tasks": [
                    {"task_id": "a", "prompt": "p", "category": "bogus_kind"}
                ],
            },
        )
        with pytest.raises(BenchmarkLoadError, match="invalid category"):
            BenchmarkSuite.from_path(path)

    def test_yaml_extension_raises_clear_error_when_pyyaml_missing(self, tmp_path):
        # If PyYAML is installed in the test environment, the loader will
        # parse YAML successfully — that's the lazy-optional contract.
        # Otherwise the loader must raise ImportError, not a load error.
        path = tmp_path / "suite.yaml"
        path.write_text("name: x\ntasks: []\n")
        try:
            import yaml  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="PyYAML"):
                BenchmarkSuite.from_path(path)
        else:
            suite = BenchmarkSuite.from_path(path)
            assert suite.name == "x"


# --------------- Runner ------------


def _solver_factory(*, success_pattern, latency_ms=10.0, regressions=0, edit_size=20):
    """Build a deterministic solver. ``success_pattern`` may be a bool or a
    callable mapping ``BenchmarkTask -> bool`` for per-task control."""
    if isinstance(success_pattern, bool):
        decide = lambda task: success_pattern  # noqa: E731
    else:
        decide = success_pattern

    def solve(task):
        return TaskAttempt(
            task_id=task.task_id,
            success=decide(task),
            latency_ms=latency_ms,
            regressions=regressions,
            edit_size=edit_size,
        )

    return solve


class TestRunModes:
    def test_runs_all_four_modes_with_fake_solvers(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        solvers = {
            "baseline": _solver_factory(success_pattern=False, latency_ms=200),
            "symbolic": _solver_factory(
                success_pattern=lambda t: t.category == "repo_local",
                latency_ms=300,
            ),
            "search": _solver_factory(
                success_pattern=lambda t: t.category != "external_slice",
                latency_ms=900,
                regressions=1,
            ),
            "search_sft": _solver_factory(success_pattern=True, latency_ms=120),
        }
        reports = run_modes(suite, solvers, holdout_name="full")
        assert set(reports.keys()) == {"baseline", "symbolic", "search", "search_sft"}
        for r in reports.values():
            assert r.n == 3
            assert r.holdout_name == "full"

    def test_completion_rate_computed(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        solvers = {
            "baseline": _solver_factory(success_pattern=False),  # 0/3
            "symbolic": _solver_factory(
                success_pattern=lambda t: t.category == "repo_local"
            ),  # 1/3
            "search_sft": _solver_factory(success_pattern=True),  # 3/3
        }
        reports = run_modes(suite, solvers)
        assert reports["baseline"].completion_rate == 0.0
        assert reports["symbolic"].completion_rate == pytest.approx(1 / 3)
        assert reports["search_sft"].completion_rate == 1.0

    def test_mean_latency_and_edit_size(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        solvers = {
            "baseline": _solver_factory(
                success_pattern=False, latency_ms=100, edit_size=10
            ),
            "search": _solver_factory(
                success_pattern=True, latency_ms=400, edit_size=80
            ),
        }
        reports = run_modes(suite, solvers)
        assert reports["baseline"].mean_latency_ms == pytest.approx(100.0)
        assert reports["search"].mean_latency_ms == pytest.approx(400.0)
        assert reports["baseline"].mean_edit_size == pytest.approx(10.0)
        assert reports["search"].mean_edit_size == pytest.approx(80.0)

    def test_regression_rate_computed(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        solvers = {
            "search": _solver_factory(success_pattern=True, regressions=2),
        }
        reports = run_modes(suite, solvers)
        # 3 tasks × 2 regressions each = 6, divided by 3 tasks = 2.0
        assert reports["search"].regression_rate == pytest.approx(2.0)

    def test_split_filter_limits_tasks_evaluated(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        called_with: list[str] = []

        def solver(task):
            called_with.append(task.task_id)
            return TaskAttempt(task_id=task.task_id, success=True, latency_ms=1.0)

        run_modes(suite, {"baseline": solver}, split="valid")
        assert called_with == ["repo-1"]

    def test_unknown_mode_rejected(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        with pytest.raises(ValueError, match="Unknown evaluation mode"):
            run_modes(suite, {"not_a_mode": _solver_factory(success_pattern=True)})

    def test_empty_suite_yields_empty_reports(self):
        suite = BenchmarkSuite(name="empty", tasks=[])
        reports = run_modes(suite, {"baseline": _solver_factory(success_pattern=True)})
        r = reports["baseline"]
        assert r.n == 0
        assert r.completion_rate == 0.0
        assert r.regression_rate == 0.0
        assert r.mean_latency_ms == 0.0


# --------------- EvaluationResult bridge + PromotionGate ------------


class TestEvaluationResultBridge:
    def test_to_evaluation_result_carries_metrics(self):
        report = ModeReport(
            mode="search_sft",
            holdout_name="primary",
            attempts=[
                TaskAttempt(task_id="t1", success=True, latency_ms=100, regressions=0, edit_size=20),
                TaskAttempt(task_id="t2", success=True, latency_ms=200, regressions=1, edit_size=80),
                TaskAttempt(task_id="t3", success=False, latency_ms=150, regressions=2, edit_size=40),
            ],
        )
        ev = report.to_evaluation_result()
        assert isinstance(ev, EvaluationResult)
        assert ev.mode == "search_sft"
        assert ev.holdout_name == "primary"
        assert ev.completion_rate == pytest.approx(2 / 3)
        assert ev.pass_rate == pytest.approx(2 / 3)
        assert ev.regression_rate == pytest.approx(1.0)  # (0+1+2)/3
        assert ev.latency_ms == pytest.approx(150.0)  # (100+200+150)/3
        assert ev.mean_edit_size == pytest.approx((20 + 80 + 40) / 3)
        assert ev.test_scores == {"n": 3, "completed": 2}

    def test_promotion_gate_consumes_evaluation_results(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        # symbolic completes 1/3 tasks; search_sft completes all 3.
        solvers = {
            "symbolic": _solver_factory(
                success_pattern=lambda t: t.category == "repo_local",
                latency_ms=400,
            ),
            "search_sft": _solver_factory(success_pattern=True, latency_ms=800),
        }
        reports = run_modes(suite, solvers, holdout_name="primary")
        new_eval, current_eval = compare_modes(
            reports, new_mode="search_sft", current_mode="symbolic"
        )
        gate = PromotionGate(
            primary_lift_threshold=0.05, latency_bound_ms=30_000
        )
        decision = gate.evaluate(new_eval, current_eval, smoke_passed=True)
        assert decision.decision == "promote"
        assert decision.primary_lift > 0.05

    def test_promotion_gate_rejects_when_no_lift(self, tmp_path):
        suite = BenchmarkSuite.from_path(_write_suite(tmp_path, _suite_dict()))
        solvers = {
            "symbolic": _solver_factory(success_pattern=True, latency_ms=400),
            "search_sft": _solver_factory(success_pattern=True, latency_ms=400),
        }
        reports = run_modes(suite, solvers)
        new_eval, current_eval = compare_modes(
            reports, new_mode="search_sft", current_mode="symbolic"
        )
        gate = PromotionGate(primary_lift_threshold=0.05)
        decision = gate.evaluate(new_eval, current_eval)
        assert decision.decision == "reject"
        assert "lift" in decision.reason.lower()

    def test_compare_modes_raises_for_missing_mode(self):
        suite = BenchmarkSuite(name="x", tasks=[])
        reports = run_modes(suite, {"baseline": _solver_factory(success_pattern=True)})
        with pytest.raises(KeyError):
            compare_modes(reports, new_mode="search_sft", current_mode="baseline")
