from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cl_layer.episode.recorder import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.pi_mono import (  # noqa: E402
    ContextBuilder,
    PiCliRunner,
    PiProcessResult,
    append_session_episode,
    import_session_jsonl,
    load_session_lines,
    session_to_episode,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _rich_session_rows() -> list[dict]:
    return [
        {
            "type": "session",
            "version": 3,
            "id": "sess-123",
            "timestamp": "2026-04-20T10:00:00.000Z",
            "cwd": "/repo/project",
            "parentSession": "/tmp/parent.jsonl",
            "provider": "anthropic",
            "modelId": "claude-sonnet-4-5",
        },
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": "2026-04-20T10:00:01.000Z",
            "message": {"role": "user", "content": "Fix the failing tests", "timestamp": 1776679201000},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will run the tests."},
                    {"type": "toolCall", "id": "call-bash", "name": "bash", "arguments": {"command": "pytest -q"}},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "usage": {"input": 100, "output": 20},
                "stopReason": "toolUse",
                "timestamp": 1776679202000,
            },
        },
        {
            "type": "message",
            "id": "r1",
            "parentId": "a1",
            "timestamp": "2026-04-20T10:00:03.000Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-bash",
                "toolName": "bash",
                "content": [{"type": "text", "text": "1 failed\n\nCommand exited with code 1"}],
                "isError": True,
                "timestamp": 1776679203000,
            },
        },
        {
            "type": "compaction",
            "id": "c1",
            "parentId": "r1",
            "timestamp": "2026-04-20T10:00:04.000Z",
            "summary": "Earlier context was compacted.",
            "firstKeptEntryId": "u1",
            "tokensBefore": 12000,
        },
        {
            "type": "message",
            "id": "a2",
            "parentId": "c1",
            "timestamp": "2026-04-20T10:00:05.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "call-read", "name": "read", "arguments": {"path": "src/app.py"}}
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "usage": {"input": 50, "output": 5},
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "id": "r2",
            "parentId": "a2",
            "timestamp": "2026-04-20T10:00:06.000Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-read",
                "toolName": "read",
                "content": [{"type": "text", "text": "def old(): pass"}],
                "isError": False,
            },
        },
        {
            "type": "message",
            "id": "a3",
            "parentId": "r2",
            "timestamp": "2026-04-20T10:00:07.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-edit",
                        "name": "edit",
                        "arguments": {
                            "path": "src/app.py",
                            "edits": [{"oldText": "def old(): pass", "newText": "def new(): pass"}],
                        },
                    }
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "usage": {"input": 20, "output": 10},
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "id": "r3",
            "parentId": "a3",
            "timestamp": "2026-04-20T10:00:08.000Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-edit",
                "toolName": "edit",
                "content": [{"type": "text", "text": "Successfully replaced 1 block(s) in src/app.py."}],
                "details": {"diff": "--- a/src/app.py\n+++ b/src/app.py\n@@\n-def old(): pass\n+def new(): pass"},
                "isError": False,
            },
        },
        {
            "type": "message",
            "id": "a4",
            "parentId": "r3",
            "timestamp": "2026-04-20T10:00:09.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-write",
                        "name": "write",
                        "arguments": {"path": "docs/notes.md", "content": "notes"},
                    }
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "usage": {"input": 10, "output": 5},
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "id": "r4",
            "parentId": "a4",
            "timestamp": "2026-04-20T10:00:10.000Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-write",
                "toolName": "write",
                "content": [{"type": "text", "text": "Successfully wrote 5 bytes to docs/notes.md"}],
                "isError": False,
            },
        },
        {
            "type": "message",
            "id": "a5",
            "parentId": "r4",
            "timestamp": "2026-04-20T10:00:11.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I updated src/app.py and docs/notes.md."}],
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "usage": {"input": 5, "output": 5},
                "stopReason": "stop",
            },
        },
    ]


def test_pi_session_import_maps_events_metadata_and_outcome(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    _write_jsonl(session_path, _rich_session_rows())

    episode = import_session_jsonl(
        session_path,
        task_id="task-pi",
        task_domain="python",
        task_description="Fix the failing tests",
        mode="baseline",
    )

    assert episode.thread_id == "sess-123"
    assert episode.run_id == "pi-mono:sess-123:a5"
    assert episode.repo_path == "/repo/project"
    assert episode.parent_episode_id == "/tmp/parent.jsonl"
    assert episode.base_model_id == "anthropic/claude-sonnet-4-5"
    assert episode.cost_tokens_prompt == 185
    assert episode.cost_tokens_completion == 45
    assert episode.outcome.status == "partial"
    assert episode.outcome.tests_passed is False
    assert episode.outcome.files_touched == ["docs/notes.md", "src/app.py"]
    assert episode.outcome.final_response == "I updated src/app.py and docs/notes.md."
    assert episode.patch_text and "def new" in episode.patch_text
    assert episode.patch_hash and episode.patch_hash.startswith("sha256:")

    command_events = [event for event in episode.events if event.kind == "command_execution"]
    assert len(command_events) == 1
    assert command_events[0].payload["command"] == "pytest -q"
    assert command_events[0].payload["exit_code"] == 1

    file_events = [event for event in episode.events if event.kind == "file_change"]
    operations = {(event.payload["operation"], event.payload["path"]) for event in file_events}
    assert operations == {("read", "src/app.py"), ("edit", "src/app.py"), ("write", "docs/notes.md")}

    summary_events = [
        event for event in episode.events if event.kind == "agent_message" and event.payload["role"] == "compactionSummary"
    ]
    assert summary_events[0].payload["tokens_before"] == 12000


def test_pi_read_file_change_does_not_count_as_file_touched() -> None:
    rows = [
        {"type": "session", "version": 3, "id": "sess-read-only", "timestamp": "2026-04-20T10:00:00.000Z", "cwd": "/repo"},
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": "2026-04-20T10:00:01.000Z",
            "message": {"role": "user", "content": "inspect README"},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "call-read", "name": "read", "arguments": {"path": "README.md"}}
                ],
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "id": "r1",
            "parentId": "a1",
            "timestamp": "2026-04-20T10:00:03.000Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-read",
                "toolName": "read",
                "content": [{"type": "text", "text": "# Project"}],
                "isError": False,
            },
        },
    ]

    episode = session_to_episode(
        load_session_lines(json.dumps(row) for row in rows),
        task_id="read-only",
        task_domain="docs",
        task_description="Inspect README",
    )

    file_events = [event for event in episode.events if event.kind == "file_change"]
    assert [(event.payload["operation"], event.payload["path"], event.payload["mutating"]) for event in file_events] == [
        ("read", "README.md", False)
    ]
    assert episode.outcome.files_touched == []


def test_pi_branch_selection_imports_selected_leaf_only() -> None:
    rows = [
        {"type": "session", "version": 3, "id": "sess-branch", "timestamp": "2026-04-20T10:00:00.000Z", "cwd": "/repo"},
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": "2026-04-20T10:00:01.000Z",
            "message": {"role": "user", "content": "start"},
        },
        {
            "type": "message",
            "id": "main",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:02.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "main branch"}], "stopReason": "stop"},
        },
        {
            "type": "message",
            "id": "alt",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:03.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "alternate branch"}],
                "stopReason": "stop",
            },
        },
    ]
    session = load_session_lines(json.dumps(row) for row in rows)

    selected = session_to_episode(
        session,
        task_id="branch",
        task_domain="test",
        task_description="branch",
        branch_leaf_id="main",
    )
    default_leaf = session_to_episode(session, task_id="branch", task_domain="test", task_description="branch")

    selected_texts = [event.payload.get("text") for event in selected.events if event.kind == "agent_message"]
    default_texts = [event.payload.get("text") for event in default_leaf.events if event.kind == "agent_message"]
    assert "main branch" in selected_texts
    assert "alternate branch" not in selected_texts
    assert "alternate branch" in default_texts


def test_pi_branch_selection_handles_idless_non_message_entries() -> None:
    rows = [
        {"type": "session", "version": 3, "id": "sess-mixed", "timestamp": "2026-04-20T10:00:00.000Z", "cwd": "/repo"},
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": "2026-04-20T10:00:01.000Z",
            "message": {"role": "user", "content": "start"},
        },
        {
            "type": "message",
            "id": "main",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:02.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "main branch"}], "stopReason": "stop"},
        },
        {
            "type": "compaction",
            "parentId": "main",
            "timestamp": "2026-04-20T10:00:03.000Z",
            "summary": "Idless compaction from a mixed real-world session.",
            "firstKeptEntryId": "u1",
            "tokensBefore": 1000,
        },
        {
            "type": "message",
            "id": "alt",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:04.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "alternate branch"}],
                "stopReason": "stop",
            },
        },
    ]
    session = load_session_lines(json.dumps(row) for row in rows)

    selected = session_to_episode(
        session,
        task_id="mixed-branch",
        task_domain="test",
        task_description="branch",
        branch_leaf_id="main",
    )

    selected_texts = [event.payload.get("text") for event in selected.events if event.kind == "agent_message"]
    assert "main branch" in selected_texts
    assert "alternate branch" not in selected_texts


def test_pi_loader_tolerates_malformed_lines_and_bash_execution() -> None:
    lines = [
        json.dumps({"type": "session", "id": "sess-bash", "timestamp": "2026-04-20T10:00:00.000Z", "cwd": "/repo"}),
        "{not json",
        json.dumps(
            {
                "type": "message",
                "id": "b1",
                "parentId": None,
                "timestamp": "2026-04-20T10:00:01.000Z",
                "message": {
                    "role": "bashExecution",
                    "command": "npm run check",
                    "output": "ok",
                    "exitCode": 0,
                    "cancelled": False,
                    "truncated": False,
                },
            }
        ),
    ]

    session = load_session_lines(lines)
    episode = session_to_episode(
        session,
        task_id="bash",
        task_domain="node",
        task_description="Run check",
    )

    assert len(session.malformed_lines) == 1
    assert episode.outcome.status == "completed"
    assert episode.outcome.tests_passed is True
    assert episode.events[0].kind == "command_execution"
    assert episode.events[0].payload["source"] == "bashExecution"


def test_pi_error_stop_derives_failed_outcome() -> None:
    rows = [
        {"type": "session", "id": "sess-error", "timestamp": "2026-04-20T10:00:00.000Z", "cwd": "/repo"},
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": "2026-04-20T10:00:01.000Z",
            "message": {"role": "user", "content": "do it"},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Provider failed."}],
                "provider": "openai",
                "model": "gpt-5.1-codex",
                "stopReason": "error",
                "errorMessage": "upstream unavailable",
            },
        },
    ]

    episode = session_to_episode(
        load_session_lines(json.dumps(row) for row in rows),
        task_id="error",
        task_domain="test",
        task_description="error",
    )

    assert episode.outcome.status == "failed"
    assert episode.outcome.escalation_reason == "upstream unavailable"


def test_pi_missing_header_timestamp_does_not_pollute_started_at() -> None:
    rows = [
        {"type": "session", "id": "sess-no-header-time", "cwd": "/repo"},
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": "2026-04-20T10:00:01.000Z",
            "message": {"role": "user", "content": "hello"},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "u1",
            "timestamp": "2026-04-20T10:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "stopReason": "stop",
            },
        },
    ]

    episode = session_to_episode(
        load_session_lines(json.dumps(row) for row in rows),
        task_id="no-header-time",
        task_domain="test",
        task_description="missing header timestamp",
    )

    assert episode.started_at == datetime(2026, 4, 20, 10, 0, 1, tzinfo=timezone.utc)
    assert episode.ended_at == datetime(2026, 4, 20, 10, 0, 2, tzinfo=timezone.utc)


def test_pi_numeric_header_timestamp_is_parsed_as_unix_ms() -> None:
    header_ms = 1776679200000
    rows = [
        {"type": "session", "id": "sess-numeric-header-time", "timestamp": header_ms, "cwd": "/repo"},
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": header_ms + 1000,
            "message": {"role": "user", "content": "hello"},
        },
    ]

    episode = session_to_episode(
        load_session_lines(json.dumps(row) for row in rows),
        task_id="numeric-header-time",
        task_domain="test",
        task_description="numeric header timestamp",
    )

    assert episode.started_at == datetime.fromtimestamp(header_ms / 1000, tz=timezone.utc)
    assert episode.ended_at == datetime.fromtimestamp((header_ms + 1000) / 1000, tz=timezone.utc)


def test_pi_context_builder_modes(tmp_path: Path) -> None:
    (tmp_path / "PROGRAM.md").write_text("Remember the repo invariant.", encoding="utf-8")
    (tmp_path / "SKILLS.md").write_text("Use the fast test target.", encoding="utf-8")

    builder = ContextBuilder(tmp_path)
    baseline = builder.build("task", mode="baseline", cwd="/repo")
    integrated = builder.build("task", mode="integrated", cwd="/repo")

    assert baseline.append_system_prompt is None
    assert baseline.cli_args() == ["--no-session"]
    assert integrated.append_system_prompt is not None
    assert "PROGRAM.md" in integrated.append_system_prompt
    assert "SKILLS.md" in integrated.append_system_prompt
    assert "--no-session" not in integrated.cli_args()
    assert integrated.cli_args()[0] == "--append-system-prompt"


def test_pi_cli_runner_uses_injectable_runner_and_mode_flags(tmp_path: Path) -> None:
    (tmp_path / "PROGRAM.md").write_text("Prior context.", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_runner(args, cwd, env):
        calls.append(list(args))
        return PiProcessResult(args=list(args), returncode=0, stdout="done", stderr="")

    runner = PiCliRunner(pi_executable="pi-dev", artifacts_dir=tmp_path, command_runner=fake_runner)
    baseline = runner.run("do task", mode="baseline", cwd="/repo", model="anthropic/claude")
    integrated = runner.run("do task", mode="integrated", cwd="/repo", output_mode="json")

    assert baseline.returncode == 0
    assert "--no-session" in baseline.args
    assert "--model" in baseline.args
    assert "--append-system-prompt" in integrated.args
    assert "--no-session" not in integrated.args
    assert calls == [baseline.args, integrated.args]


def test_pi_append_session_episode_roundtrips_with_recorder(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(session_path, _rich_session_rows())

    episode = append_session_episode(
        session_path,
        episodes_path,
        task_id="task-pi",
        task_domain="python",
        task_description="Fix the failing tests",
    )
    loaded = EpisodeRecorder(episodes_path).load_all()

    assert len(loaded) == 1
    assert loaded[0].episode_id == episode.episode_id
    assert loaded[0].agent_surface == "pi_mono"
