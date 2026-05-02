from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cl_layer.episode.recorder import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.swe_agent import (  # noqa: E402
    ContextBuilder,
    SWEAgentProcessResult,
    SWEAgentRunner,
    SWEAgentTrajectory,
    append_trajectory_episode,
    import_trajectory,
    load_trajectory,
    map_swe_agent_trajectory,
    paths_from_unified_diff,
    trajectory_to_episode,
)


PATCH = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
-print("old")
+print("new")
"""


def _success_traj() -> dict:
    return {
        "environment": "swe_main",
        "trajectory": [
            {
                "response": "I will run the focused tests.\n\n```\npytest -q tests/test_app.py\n```",
                "thought": "Run focused tests first.",
                "action": "pytest -q tests/test_app.py",
                "observation": "1 passed in 0.10s",
                "execution_time": 1.25,
                "state": {"working_dir": "/repo", "open_file": "n/a"},
            },
            {
                "response": "I will edit the open file.",
                "thought": "Patch the behavior.",
                "action": "edit 1:1\nprint(\"new\")\nend_of_edit",
                "observation": "Edit successful.",
                "execution_time": 0.25,
                "state": {"working_dir": "/repo", "open_file": "/repo/src/app.py"},
            },
            {
                "response": "The fix is ready; I will submit.",
                "thought": "Submit the final patch.",
                "action": "submit",
                "observation": PATCH,
                "execution_time": 0.5,
                "state": {"working_dir": "/repo", "open_file": "/repo/src/app.py"},
            },
        ],
        "history": [
            {"role": "system", "content": "System prompt", "agent": "main", "message_type": "system_prompt"},
            {"role": "user", "content": "Fix app.py", "agent": "main", "message_type": "observation"},
            {
                "role": "assistant",
                "content": "I will edit.",
                "thought": "Patch the behavior.",
                "action": "edit 1:1\nprint(\"new\")\nend_of_edit",
                "agent": "main",
                "message_type": "action",
            },
        ],
        "info": {
            "exit_status": "submitted",
            "submission": PATCH,
            "model": "gpt-test",
            "started_at": "2026-04-20T10:00:00+00:00",
            "ended_at": "2026-04-20T10:00:02+00:00",
            "tests_passed": True,
            "verification_summary": "Verifier passed focused tests.",
            "verification_steps": [{"name": "pytest", "passed": True}],
            "verification_score": 1.0,
            "model_stats": {"tokens_sent": 100, "tokens_received": 20, "api_calls": 3},
        },
    }


def _failure_traj() -> dict:
    return {
        "environment": "swe_main",
        "trajectory": [
            {
                "response": "I will run tests.",
                "thought": "Check the failure.",
                "action": "pytest -q",
                "observation": "FAILED tests/test_app.py::test_health\nexit code: 1",
                "execution_time": 2.0,
                "state": "{\"working_dir\": \"/repo\", \"open_file\": \"n/a\"}",
            }
        ],
        "history": [
            {"role": "system", "content": "System prompt", "agent": "main"},
            {"role": "user", "content": "Fix failing tests", "agent": "main"},
        ],
        "info": {
            "exit_status": "exit_cost",
            "tests_passed": False,
            "verification_summary": "Verifier reported failing tests.",
            "verification_failures": ["tests/test_app.py::test_health"],
            "model_stats": {"tokens_sent": 50, "tokens_received": 8},
        },
    }


def test_swe_agent_success_trajectory_maps_patch_commands_verifier_and_roundtrip(tmp_path: Path) -> None:
    traj_path = tmp_path / "success.traj"
    traj_path.write_text(json.dumps(_success_traj()), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text('agent:\n  model:\n    name: "configured-model"\n', encoding="utf-8")

    episode = import_trajectory(
        traj_path,
        task_id="swe-success",
        task_domain="python",
        mode="baseline",
    )

    assert episode.agent_surface == "swe_agent"
    assert episode.task_id == "swe-success"
    assert episode.base_model_id == "gpt-test"
    assert episode.repo_path == "/repo"
    assert episode.outcome.status == "completed"
    assert episode.outcome.tests_passed is True
    assert episode.outcome.verification_summary == "Verifier passed focused tests."
    assert episode.outcome.files_touched == ["src/app.py"]
    assert episode.reward is None
    assert episode.patch_text == PATCH
    assert episode.patch_hash and episode.patch_hash.startswith("sha256:")
    assert episode.verification_steps == [{"name": "pytest", "passed": True}]
    assert episode.verification_score == 1.0
    assert episode.cost_tokens_prompt == 100
    assert episode.cost_tokens_completion == 20
    assert episode.latency_ms == 2000
    assert episode.stdout_excerpt == "1 passed in 0.10s"
    assert "diff --git" not in episode.stdout_excerpt

    command_events = [event for event in episode.events if event.kind == "command_execution"]
    assert [event.payload["command"] for event in command_events] == [
        "pytest -q tests/test_app.py",
        'edit 1:1\nprint("new")\nend_of_edit',
        "submit",
    ]
    assert command_events[0].payload["test_passed"] is True
    assert command_events[0].timestamp < command_events[1].timestamp < command_events[2].timestamp
    assert max(event.timestamp for event in episode.events) <= episode.ended_at

    file_events = [event for event in episode.events if event.kind == "file_change"]
    assert any(event.payload["source"] == "swe_agent_action" for event in file_events)
    submission_event = next(event for event in file_events if event.payload["source"] == "swe_agent_submission")
    assert submission_event.payload["patch_hash"] == episode.patch_hash
    assert submission_event.payload["paths"] == ["src/app.py"]

    assert any(event.kind == "evaluation_result" for event in episode.events)
    metadata = next(
        event for event in episode.events if event.payload.get("swe_agent_event_type") == "run_metadata"
    )
    assert metadata.payload["exit_status"] == "submitted"
    assert metadata.payload["config"]["type"] == "text"
    assert "excerpt" not in metadata.payload["config"]

    episodes_path = tmp_path / "episodes.jsonl"
    appended = append_trajectory_episode(
        traj_path,
        episodes_path,
        task_id="swe-success",
        task_domain="python",
    )
    loaded = EpisodeRecorder(episodes_path).load_all()
    assert len(loaded) == 1
    assert loaded[0].episode_id == appended.episode_id
    assert loaded[0].patch_text == PATCH
    assert loaded[0].patch_hash == appended.patch_hash


def test_swe_agent_failure_trajectory_keeps_failed_verifier_and_no_reward(tmp_path: Path) -> None:
    traj_path = tmp_path / "failure.traj"
    traj_path.write_text(json.dumps(_failure_traj()), encoding="utf-8")

    episode = import_trajectory(
        traj_path,
        task_id="swe-failure",
        task_domain="python",
        mode="baseline",
    )

    assert episode.outcome.status == "failed"
    assert episode.outcome.tests_passed is False
    assert episode.outcome.verification_summary == "Verifier reported failing tests."
    assert episode.outcome.escalation_reason == "latest test-like command failed"
    assert episode.verification_failures == ["tests/test_app.py::test_health"]
    assert episode.patch_text is None
    assert episode.patch_hash is None
    assert episode.reward is None
    assert episode.test_trace == [
        {
            "command": "pytest -q",
            "exit_code": 1,
            "passed": False,
            "source": "swe_agent_trajectory",
            "step_index": 0,
        }
    ]
    assert episode.stderr_excerpt and "FAILED" in episode.stderr_excerpt


def test_swe_agent_mapper_uses_submit_observation_when_info_submission_missing() -> None:
    raw = _success_traj()
    raw["info"] = {"exit_status": "submitted"}

    mapped = map_swe_agent_trajectory(
        raw,
        timestamp=episode_time(),
        source_path=None,
        source_config=None,
    )

    assert mapped.patch_text == PATCH
    assert mapped.patch_hash and mapped.patch_hash.startswith("sha256:")
    assert mapped.outcome.files_touched == ["src/app.py"]


def test_swe_agent_context_builder_and_runner_preview_write_inspectable_config(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "PROGRAM.md").write_text("Prefer focused pytest runs with {{ braces }}.", encoding="utf-8")
    (artifacts / "SKILLS.md").write_text("Use pathlib for paths.", encoding="utf-8")

    builder = ContextBuilder(artifacts)
    baseline = builder.build("Fix parser", mode="baseline", cwd="/repo")
    integrated = builder.build("Fix parser", mode="integrated", cwd="/repo")

    assert baseline.injected_artifacts == []
    assert "strategy_template" not in baseline.config_overlay["agent"]["templates"]
    assert "strategy_template" not in baseline.config_text()
    assert integrated.injected_artifacts == ["PROGRAM.md", "SKILLS.md"]
    assert "strategy_template" in integrated.config_overlay["agent"]["templates"]
    assert "PROGRAM.md" in integrated.config_text()
    assert "{% raw %}" in integrated.config_text()
    assert "Prefer focused pytest" in integrated.config_text()

    seen: dict[str, object] = {}

    def fake_runner(args, cwd, env, timeout_seconds):
        seen["args"] = list(args)
        seen["cwd"] = cwd
        seen["env"] = dict(env or {})
        seen["timeout_seconds"] = timeout_seconds
        return SWEAgentProcessResult(args=list(args), returncode=0, stdout="ok", stderr="")

    runner = SWEAgentRunner(
        artifacts_dir=artifacts,
        run_artifacts_dir=tmp_path / "captures",
        base_config="/swe-agent/config/default.yaml",
        timeout_seconds=30,
        command_runner=fake_runner,
    )
    preview = runner.preview(
        "Fix parser",
        task_id="parser-task",
        mode="integrated",
        cwd="/repo",
        repo_path="/repo",
        model="gpt-4o",
        output_dir=tmp_path / "out",
    )

    assert Path(preview.config_path).exists()
    assert Path(preview.problem_statement_path).read_text(encoding="utf-8") == "Fix parser"
    assert "--config" in preview.args
    assert "/swe-agent/config/default.yaml" in preview.args
    assert f"--problem_statement.path={preview.problem_statement_path}" in preview.args
    assert "--problem_statement.id=parser-task" in preview.args
    assert "--env.repo.type=local" in preview.args
    assert "--env.repo.path=/repo" in preview.args
    assert "--agent.model.name=gpt-4o" in preview.args

    result = runner.run(
        "Fix parser",
        task_id="parser-task",
        mode="baseline",
        cwd="/repo",
        repo_path="/repo",
        env={"SWE_AGENT_TEST": "1"},
    )
    assert result.process.returncode == 0
    assert seen["cwd"] == "/repo"
    assert seen["env"] == {"SWE_AGENT_TEST": "1"}
    assert seen["timeout_seconds"] == 30


def test_swe_agent_paths_from_unified_diff_handles_multiple_files() -> None:
    patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@
diff --git a/docs/old.md b/docs/new.md
--- a/docs/old.md
+++ b/docs/new.md
@@
"""
    assert paths_from_unified_diff(patch) == ["docs/new.md", "src/a.py"]


def test_swe_agent_submitted_status_wins_over_failed_test_evidence(tmp_path: Path) -> None:
    raw = _success_traj()
    raw["trajectory"][0]["observation"] = "FAILED tests/test_app.py::test_app\nexit code: 1"
    raw["info"]["tests_passed"] = False
    traj_path = tmp_path / "submitted-with-failed-test.traj"
    traj_path.write_text(json.dumps(raw), encoding="utf-8")

    episode = import_trajectory(
        traj_path,
        task_id="submitted-with-failed-test",
        task_domain="python",
    )

    assert episode.outcome.status == "completed"
    assert episode.outcome.tests_passed is False
    assert episode.outcome.escalation_reason is None


def test_swe_agent_submitted_variant_is_partial_when_not_plain_success() -> None:
    raw = _success_traj()
    raw["info"]["exit_status"] = "submitted (exit_cost)"
    raw["info"].pop("tests_passed", None)

    mapped = map_swe_agent_trajectory(
        raw,
        timestamp=episode_time(),
        source_path=None,
        source_config=None,
    )

    assert mapped.outcome.status == "partial"
    assert mapped.outcome.escalation_reason == "SWE-agent exit_status=submitted (exit_cost)"


def test_swe_agent_failed_tests_still_fail_non_success_exit_status(tmp_path: Path) -> None:
    partial_raw = _failure_traj()
    partial_raw["info"]["submission"] = PATCH
    partial_path = tmp_path / "partial.traj"
    partial_path.write_text(json.dumps(partial_raw), encoding="utf-8")

    failed_path = tmp_path / "failed.traj"
    failed_path.write_text(json.dumps(_failure_traj()), encoding="utf-8")

    partial = import_trajectory(partial_path, task_id="partial", task_domain="python")
    failed = import_trajectory(failed_path, task_id="failed", task_domain="python")

    assert partial.outcome.status == "partial"
    assert partial.outcome.escalation_reason == "latest test-like command failed"
    assert failed.outcome.status == "failed"


def test_swe_agent_model_from_yaml_config_is_scoped_to_agent_model_name(tmp_path: Path) -> None:
    raw = _success_traj()
    raw["info"].pop("model")
    traj_path = tmp_path / "model-scope.traj"
    traj_path.write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                "tools:",
                "  - name: my_tool_set",
                "agent:",
                "  templates:",
                "    name: template_name",
                "  model:",
                "    name: gpt-4o",
            ]
        ),
        encoding="utf-8",
    )

    episode = import_trajectory(traj_path, task_id="model-scope", task_domain="python")

    assert episode.base_model_id == "gpt-4o"


def test_swe_agent_model_from_yaml_config_ignores_unrelated_name_keys() -> None:
    raw = _success_traj()
    raw["info"].pop("model")

    mapped = map_swe_agent_trajectory(
        raw,
        timestamp=episode_time(),
        source_path=None,
        source_config="tools:\n  - name: my_tool_set\nenv:\n  name: swe_main\n",
    )

    assert mapped.base_model_id is None


def test_swe_agent_metadata_does_not_store_config_excerpt_or_secrets() -> None:
    raw = _success_traj()
    raw["info"].pop("model")
    secret_config = "\n".join(
        [
            "agent:",
            "  model:",
            "    name: gpt-4o",
            "    api_key: sk-secret-value",
            "OPENAI_API_KEY: another-secret",
        ]
    )

    mapped = map_swe_agent_trajectory(
        raw,
        timestamp=episode_time(),
        source_path="/runs/success.traj",
        source_config=secret_config,
        source_config_path="/runs/config.yaml",
    )

    assert mapped.base_model_id == "gpt-4o"
    metadata = next(
        event for event in mapped.events if event.payload.get("swe_agent_event_type") == "run_metadata"
    )
    assert metadata.payload["config"] == {
        "type": "text",
        "chars": len(secret_config),
        "path": "/runs/config.yaml",
        "redacted": True,
    }
    payload_text = json.dumps([event.payload for event in mapped.events], sort_keys=True)
    assert "sk-secret-value" not in payload_text
    assert "another-secret" not in payload_text
    assert "OPENAI_API_KEY" not in payload_text


def test_swe_agent_event_timestamps_without_execution_time_are_deterministic() -> None:
    raw = _success_traj()
    for step in raw["trajectory"]:
        step.pop("execution_time", None)

    mapped = map_swe_agent_trajectory(
        raw,
        timestamp=episode_time(),
        source_path=None,
        source_config=None,
    )

    command_events = [event for event in mapped.events if event.kind == "command_execution"]
    assert command_events
    assert {event.timestamp for event in command_events} == {episode_time()}


def test_swe_agent_stdout_excerpt_prefers_last_test_observation_not_submit_diff() -> None:
    raw = _success_traj()
    raw["trajectory"].insert(
        2,
        {
            "response": "Run final focused test.",
            "thought": "Verify the fix before submit.",
            "action": "python -m pytest -q tests/test_app.py",
            "observation": "2 passed in 0.22s",
            "execution_time": 0.2,
            "state": {"working_dir": "/repo", "open_file": "/repo/src/app.py"},
        },
    )

    mapped = map_swe_agent_trajectory(
        raw,
        timestamp=episode_time(),
        source_path=None,
        source_config=None,
    )

    assert mapped.stdout_excerpt == "2 passed in 0.22s"
    assert mapped.patch_text == PATCH
    assert "diff --git" not in mapped.stdout_excerpt


def test_swe_agent_working_dir_fallback_clamps_to_repo_root_and_normalizes_paths() -> None:
    raw = {
        "environment": "swe_main",
        "trajectory": [
            {
                "response": "Edit a test helper.",
                "thought": "Patch file outside the current subdirectory.",
                "action": "edit 1:1\nfixed = True\nend_of_edit",
                "observation": "Edit successful.",
                "execution_time": 0.2,
                "state": {"working_dir": "/repo/src/models", "open_file": "/repo/tests/test_foo.py"},
            }
        ],
        "history": [{"role": "user", "content": "Fix tests"}],
        "info": {"exit_status": "submitted"},
    }

    mapped = map_swe_agent_trajectory(
        raw,
        timestamp=episode_time(),
        source_path=None,
        source_config=None,
    )

    assert mapped.repo_path == "/repo"
    assert mapped.outcome.files_touched == ["tests/test_foo.py"]
    file_change = next(event for event in mapped.events if event.kind == "file_change")
    assert file_change.payload["paths"] == ["tests/test_foo.py"]
    assert all(not path.startswith("repo/") for path in mapped.outcome.files_touched)


def test_swe_agent_stable_missing_id_uses_bounded_fingerprint() -> None:
    raw = {
        "environment": "swe_main",
        "trajectory": [],
        "history": [],
        "info": {},
        "unserializable_large_payload": object(),
    }
    entry = SWEAgentTrajectory(
        raw=raw,
        trajectory=[],
        history=[],
        info={},
        environment="swe_main",
        source_path=None,
    )

    episode = trajectory_to_episode(entry, task_domain="python")

    assert episode.task_id.startswith("missing-run-")
    assert episode.run_id.startswith("swe-agent:missing-run-")


def test_swe_agent_subprocess_runner_threads_timeout_and_returns_timeout_result(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout"), output="partial stdout", stderr="partial stderr")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = SWEAgentRunner._subprocess_runner(
        ["sweagent", "run"],
        cwd="/repo",
        env={"A": "B"},
        timeout_seconds=7,
    )

    assert seen["timeout"] == 7
    assert result.returncode == 124
    assert result.stdout == "partial stdout"
    assert "partial stderr" in result.stderr
    assert "timed out after 7 seconds" in result.stderr


def test_swe_agent_imports_checked_in_full_shape_fixture() -> None:
    fixture = ROOT / "cl-layer" / "tests" / "fixtures" / "swe_agent" / "full_swe_agent_shape.traj"

    loaded = load_trajectory(fixture)
    episode = import_trajectory(fixture, task_domain="python")

    assert loaded.environment == "swe_main"
    assert episode.task_id == "swe-agent__test-repo-i1"
    assert episode.outcome.status == "completed"
    assert episode.patch_text and "diff --git" in episode.patch_text
    assert episode.patch_hash and episode.patch_hash.startswith("sha256:")
    assert episode.outcome.files_touched == ["tests/missing_colon.py"]
    assert episode.cost_tokens_prompt == 52861
    assert episode.cost_tokens_completion == 326
    assert any(event.kind == "command_execution" for event in episode.events)


def episode_time():
    from datetime import datetime, timezone

    return datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc)
