from __future__ import annotations

import json
import sys
from pathlib import Path

from cl_layer.distill.program import render_program_md
from cl_layer.distill.skills import distill_skills, distill_warnings, render_skills_md
from cl_layer.episode.recorder import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.hermes_agent import (  # noqa: E402
    ContextBuilder,
    HermesAgentRunner,
    append_trajectory_episodes,
    import_trajectory_jsonl,
    load_trajectory_json,
    load_trajectory_jsonl,
)


def _tool_call(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def _tool_response(name: str, content: object, tool_call_id: str) -> str:
    return (
        "<tool_response>\n"
        + json.dumps({"tool_call_id": tool_call_id, "name": name, "content": content}, ensure_ascii=False)
        + "\n</tool_response>"
    )


def _completed_entry() -> dict:
    return {
        "prompt_index": 0,
        "conversations": [
            {"from": "system", "value": "Hermes trajectory system prompt with tools."},
            {"from": "human", "value": "Fix the app and keep the project lesson."},
            {
                "from": "gpt",
                "value": "<think>\nPlan briefly.\n</think>\nI will run the focused tests.\n"
                + _tool_call("terminal", {"command": "python -m pytest -q tests/test_smoke.py", "timeout": 120}),
            },
            {
                "from": "tool",
                "value": _tool_response(
                    "terminal",
                    {"output": "1 passed", "exit_code": 0, "error": None},
                    "call-terminal-ok",
                ),
            },
            {
                "from": "gpt",
                "value": "\n".join(
                    [
                        _tool_call("read_file", {"path": "src/app.py", "offset": 1, "limit": 80}),
                        _tool_call("write_file", {"path": "docs/notes.md", "content": "lesson"}),
                        _tool_call(
                            "patch",
                            {
                                "mode": "patch",
                                "patch": "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old\n+new\n*** End Patch",
                            },
                        ),
                        _tool_call("mcp_github_create_issue", {"title": "smoke", "body": "created from Hermes"}),
                        _tool_call("memory", {"action": "add", "target": "memory", "content": "Use focused smoke tests."}),
                        _tool_call("skill_view", {"name": "pytest"}),
                        _tool_call("execute_code", {"code": "print('checked')"}),
                    ]
                ),
            },
            {
                "from": "tool",
                "value": "\n".join(
                    [
                        _tool_response(
                            "read_file",
                            {"content": "1|old", "path": "src/app.py", "total_lines": 1},
                            "call-read",
                        ),
                        _tool_response(
                            "write_file",
                            {"bytes_written": 6, "error": None},
                            "call-write",
                        ),
                        _tool_response(
                            "patch",
                            {
                                "success": True,
                                "diff": "--- a/src/app.py\n+++ b/src/app.py\n@@\n-old\n+new",
                                "files_modified": ["src/app.py"],
                            },
                            "call-patch",
                        ),
                        _tool_response(
                            "mcp_github_create_issue",
                            {"success": True, "issue": 123},
                            "call-mcp",
                        ),
                        _tool_response(
                            "memory",
                            {"success": True, "target": "memory", "message": "Entry added."},
                            "call-memory",
                        ),
                        _tool_response(
                            "skill_view",
                            {"success": True, "name": "pytest", "content": "Run pytest with -q."},
                            "call-skill",
                        ),
                        _tool_response(
                            "execute_code",
                            {"status": "success", "output": "checked", "tool_calls_made": 0, "duration_seconds": 0.1},
                            "call-code",
                        ),
                    ]
                ),
            },
            {"from": "gpt", "value": "<think>\n</think>\nDone with the focused fix."},
        ],
        "metadata": {"batch_num": 2, "timestamp": "2026-04-20T10:00:00+00:00", "model": "nous/hermes-4"},
        "completed": True,
        "partial": False,
        "api_calls": 3,
        "toolsets_used": ["terminal", "file", "memory", "skills"],
    }


def _failed_entry() -> dict:
    return {
        "prompt_index": 1,
        "conversations": [
            {"from": "system", "value": "Hermes trajectory system prompt with tools."},
            {"from": "human", "value": "Fix failing tests."},
            {
                "from": "gpt",
                "value": "I will run the tests.\n" + _tool_call("terminal", {"command": "pytest -q"}),
            },
            {
                "from": "tool",
                "value": _tool_response(
                    "terminal",
                    {"output": "FAILED tests/test_app.py::test_health", "exit_code": 1, "error": None},
                    "call-terminal-fail",
                ),
            },
            {"from": "gpt", "value": "I cannot complete because the tests failed."},
        ],
        "metadata": {"batch_num": 2, "timestamp": "2026-04-20T10:05:00+00:00", "model": "nous/hermes-4"},
        "completed": False,
        "partial": False,
    }


def _final_text_failed_entry() -> dict:
    return {
        "prompt_index": 2,
        "conversations": [
            {"from": "system", "value": "Hermes trajectory system prompt with tools."},
            {"from": "human", "value": "Try the task."},
            {"from": "gpt", "value": "I cannot complete this because the repository is missing."},
        ],
        "metadata": {"batch_num": 2, "timestamp": "2026-04-20T10:06:00+00:00", "model": "nous/hermes-4"},
        "completed": True,
        "partial": False,
    }


def _partial_entry() -> dict:
    return {
        "prompt_index": 3,
        "conversations": [
            {"from": "system", "value": "Hermes trajectory system prompt with tools."},
            {"from": "human", "value": "Attempt the long task."},
            {"from": "gpt", "value": "I made partial progress before stopping."},
        ],
        "metadata": {"batch_num": 2, "timestamp": "2026-04-20T10:07:00+00:00", "model": "nous/hermes-4"},
        "completed": False,
        "partial": True,
    }


def _mixed_test_entry() -> dict:
    return {
        "prompt_index": 4,
        "conversations": [
            {"from": "system", "value": "Hermes trajectory system prompt with tools."},
            {"from": "human", "value": "Run tests until the focused suite passes."},
            {"from": "gpt", "value": _tool_call("terminal", {"command": "pytest -q"})},
            {
                "from": "tool",
                "value": _tool_response(
                    "terminal",
                    {"output": "1 failed", "exit_code": 1, "error": None},
                    "call-test-fail",
                ),
            },
            {"from": "gpt", "value": _tool_call("terminal", {"command": "python -m pytest -q tests/test_smoke.py"})},
            {
                "from": "tool",
                "value": _tool_response(
                    "terminal",
                    {"output": "1 passed", "exit_code": 0, "error": None},
                    "call-test-pass",
                ),
            },
            {"from": "gpt", "value": "Focused tests now pass."},
        ],
        "metadata": {"batch_num": 2, "timestamp": "2026-04-20T10:08:00+00:00", "model": "nous/hermes-4"},
        "completed": True,
        "partial": False,
    }


def _write_jsonl(path: Path, rows: list[dict], malformed: bool = False) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    if malformed:
        content += "{not-json\n"
    path.write_text(content, encoding="utf-8")


def test_hermes_trajectory_import_maps_tools_messages_and_outcomes(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        trajectory_path,
        [_completed_entry(), _failed_entry(), _final_text_failed_entry(), _partial_entry()],
        malformed=True,
    )

    batch = load_trajectory_jsonl(trajectory_path)
    assert len(batch.entries) == 4
    assert len(batch.malformed_lines) == 1

    episodes = import_trajectory_jsonl(
        trajectory_path,
        task_id_prefix="hermes-task",
        task_domain="python",
        mode="baseline",
    )

    completed, failed, final_text_failed, partial = episodes
    assert completed.agent_surface == "hermes_agent"
    assert completed.thread_id == "batch-2:prompt-0"
    assert completed.base_model_id == "nous/hermes-4"
    assert completed.outcome.status == "completed"
    assert completed.outcome.tests_passed is True
    assert completed.outcome.final_response == "Done with the focused fix."
    assert completed.outcome.files_touched == ["docs/notes.md", "src/app.py"]
    assert completed.patch_text and "+new" in completed.patch_text
    assert completed.patch_hash and completed.patch_hash.startswith("sha256:")

    command_events = [event for event in completed.events if event.kind == "command_execution"]
    assert {event.payload["tool_name"] for event in command_events} == {"terminal", "execute_code"}
    assert command_events[0].payload["exit_code"] == 0

    file_events = [event for event in completed.events if event.kind == "file_change"]
    assert {(event.payload["operation"], event.payload["mutating"]) for event in file_events} >= {
        ("read", False),
        ("write", True),
        ("patch", True),
    }

    mcp_events = [event for event in completed.events if event.kind == "mcp_tool_call"]
    assert mcp_events[0].payload["server"] == "github"
    assert mcp_events[0].payload["tool"] == "create_issue"

    agent_event_types = {
        event.payload.get("hermes_event_type")
        for event in completed.events
        if event.kind == "agent_message"
    }
    assert "memory_event" in agent_event_types
    assert "skill_event" in agent_event_types

    assert failed.outcome.status == "failed"
    assert failed.outcome.tests_passed is False
    assert failed.outcome.verification_summary == "Latest test-like command failed."
    assert failed.outcome.escalation_reason == "Hermes trajectory marked incomplete."
    failed_commands = [event for event in failed.events if event.kind == "command_execution"]
    assert failed_commands[0].payload["command"] == "pytest -q"
    assert failed_commands[0].payload["exit_code"] == 1

    assert final_text_failed.outcome.status == "partial"
    assert final_text_failed.outcome.escalation_reason == "final assistant response indicates failure"

    assert partial.outcome.status == "partial"
    assert partial.outcome.escalation_reason == "Hermes trajectory marked partial."


def test_hermes_load_trajectory_json_supports_pretty_sample_file(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(_completed_entry(), ensure_ascii=False, indent=2), encoding="utf-8")

    batch = load_trajectory_json(sample_path)

    assert len(batch.entries) == 1
    assert batch.entries[0].completed is True
    assert batch.entries[0].partial is False


def test_hermes_latest_test_command_sets_tests_passed_but_trace_keeps_all(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "mixed-tests.jsonl"
    _write_jsonl(trajectory_path, [_mixed_test_entry()])

    episode = import_trajectory_jsonl(
        trajectory_path,
        task_id_prefix="hermes-task",
        task_domain="python",
    )[0]

    assert episode.test_trace == [
        {"command": "pytest -q", "exit_code": 1, "source": "hermes_trajectory"},
        {"command": "python -m pytest -q tests/test_smoke.py", "exit_code": 0, "source": "hermes_trajectory"},
    ]
    assert episode.outcome.tests_passed is True
    assert episode.outcome.verification_summary == "Latest test-like command passed."


def test_hermes_append_roundtrip_and_distillation_functions(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(trajectory_path, [_completed_entry(), _failed_entry()])

    appended = append_trajectory_episodes(
        trajectory_path,
        episodes_path,
        task_id_prefix="hermes-task",
        task_domain="python",
    )
    loaded = EpisodeRecorder(episodes_path).load_all()

    assert [episode.episode_id for episode in loaded] == [episode.episode_id for episode in appended]
    warnings = distill_warnings(loaded, threshold=1)
    skills = distill_skills(loaded)
    skills_md = render_skills_md(skills)
    program_md = render_program_md("python", skills, warnings)

    assert "Domain 'python' has 1 failure" in warnings[0]
    assert "# SKILLS" in skills_md
    assert "Failure Warnings" in program_md


def test_hermes_context_builder_and_runner_are_injectable(tmp_path: Path) -> None:
    (tmp_path / "PROGRAM.md").write_text("Prefer focused tests.", encoding="utf-8")
    (tmp_path / "SKILLS.md").write_text("Use read_file before patch.", encoding="utf-8")

    builder = ContextBuilder(tmp_path)
    baseline = builder.build("task", mode="baseline", cwd="/repo")
    integrated = builder.build("task", mode="integrated", cwd="/repo")

    assert baseline.ephemeral_system_prompt is None
    assert baseline.agent_kwargs()["skip_memory"] is True
    assert baseline.agent_kwargs()["skip_context_files"] is True
    assert baseline.agent_kwargs()["persist_session"] is False
    assert integrated.ephemeral_system_prompt is not None
    assert "PROGRAM.md" in integrated.ephemeral_system_prompt
    assert "SKILLS.md" in integrated.ephemeral_system_prompt
    assert integrated.agent_kwargs()["skip_memory"] is False
    assert integrated.agent_kwargs()["persist_session"] is True

    calls: list[dict] = []

    def fake_runner(context, kwargs):
        calls.append({"context": context, "kwargs": dict(kwargs)})
        return {"completed": True, "messages": []}

    runner = HermesAgentRunner(artifacts_dir=tmp_path, conversation_runner=fake_runner)
    result = runner.run("task", mode="integrated", cwd="/repo", model="nous/hermes-4", max_iterations=2)

    assert result.result["completed"] is True
    assert calls[0]["kwargs"]["model"] == "nous/hermes-4"
    assert calls[0]["kwargs"]["max_iterations"] == 2
    assert calls[0]["kwargs"]["ephemeral_system_prompt"] == result.context.ephemeral_system_prompt
