"""
Tests for the OpenHands V1 conversation adapter.

All tests use synthetic event streams; no OpenHands runtime, Docker, app
server, or real model calls are made.  Fixtures are built inline via factory
functions that produce plain Python dicts matching the real V1 event shape.

V1 event shapes used:
  ActionEvent      — {id, timestamp, source, action: {kind, ...}, thought, ...}
  ObservationEvent — {id, timestamp, source, observation: {kind, ...}, action_id, tool_call_id}
  MessageEvent     — {id, timestamp, source, llm_message: {role, content}}
  SystemEvent      — {id, timestamp, source, system_prompt}
  ConversationStateUpdateEvent — {id, timestamp, source, kind: "ConversationStateUpdateEvent", key, value}
  ConversationErrorEvent       — {id, timestamp, source, kind: "ConversationErrorEvent", code, detail}
  ServerErrorEvent             — {id, timestamp, source, kind: "ServerErrorEvent", code, detail}
  HookExecutionEvent — {id, timestamp, source, kind: "HookExecutionEvent", hook_event_type, hook_command, success, blocked, exit_code}
  AgentErrorEvent  — {id, timestamp, source, tool_name, tool_call_id, error}
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from cl_layer.episode.recorder import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.openhands import (  # noqa: E402
    ContextBuilder,
    OpenHandsConversation,
    OpenHandsImporter,
    OpenHandsLiveRunner,
    OpenHandsRunContext,
    append_conversation_episode,
    build_conversation_zip,
    conversation_to_episode,
    import_conversation,
    load_conversation,
    load_conversation_dir,
    load_conversation_zip,
    map_conversation,
)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

_T0 = "2026-01-01T00:00:00.000Z"
_T1 = "2026-01-01T00:00:01.000Z"
_T2 = "2026-01-01T00:00:02.000Z"
_T3 = "2026-01-01T00:00:03.000Z"
_T4 = "2026-01-01T00:00:04.000Z"


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Event factory helpers
# ---------------------------------------------------------------------------

def _action_event(
    ev_id: str,
    action_type: str,
    action_fields: dict,
    *,
    thought: str = "",
    timestamp: str = _T0,
    tool_call_id: str = "tc-1",
) -> dict:
    return {
        "id": ev_id,
        "timestamp": timestamp,
        "source": "agent",
        "thought": [{"type": "text", "text": thought}] if thought else [],
        "thinking_blocks": [],
        "action": {"kind": action_type, **action_fields},
        "tool_name": action_type.lower(),
        "tool_call_id": tool_call_id,
        "llm_response_id": f"resp-{ev_id}",
        "security_risk": "LOW",
        "summary": None,
    }


def _obs_event(
    obs_id: str,
    obs_type: str,
    obs_fields: dict,
    *,
    action_id: str,
    timestamp: str = _T1,
    tool_call_id: str = "tc-1",
) -> dict:
    return {
        "id": obs_id,
        "timestamp": timestamp,
        "source": "environment",
        "tool_name": obs_type.lower(),
        "tool_call_id": tool_call_id,
        "observation": {"kind": obs_type, **obs_fields},
        "action_id": action_id,
    }


def _message_event(
    ev_id: str,
    role: str,
    text: str,
    *,
    timestamp: str = _T0,
    activated_microagents: list[str] | None = None,
) -> dict:
    return {
        "id": ev_id,
        "timestamp": timestamp,
        "source": "agent" if role != "user" else "user",
        "llm_message": {
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
        "activated_microagents": activated_microagents or [],
        "extended_content": [],
    }


def _system_event(ev_id: str, prompt: str, *, timestamp: str = _T0) -> dict:
    return {
        "id": ev_id,
        "timestamp": timestamp,
        "source": "agent",
        "system_prompt": prompt,
    }


def _state_event(ev_id: str, status: str, *, timestamp: str = _T0) -> dict:
    return {
        "id": ev_id,
        "timestamp": timestamp,
        "source": "environment",
        "kind": "ConversationStateUpdateEvent",
        "key": "execution_status",
        "value": status,
    }


def _finish_action_event(ev_id: str, message: str, *, timestamp: str = _T3) -> dict:
    return _action_event(
        ev_id,
        "FinishAction",
        {"message": message},
        timestamp=timestamp,
    )


def _error_event(ev_id: str, error: str, *, timestamp: str = _T3) -> dict:
    return {
        "id": ev_id,
        "timestamp": timestamp,
        "source": "agent",
        "tool_name": "bash",
        "tool_call_id": f"tc-{ev_id}",
        "error": error,
    }


# ---------------------------------------------------------------------------
# Full conversation factories
# ---------------------------------------------------------------------------

def _bash_success_events() -> list[dict]:
    """A successful bash command run."""
    act = _action_event(
        "act-1", "ExecuteBashAction",
        {"command": "pytest -q tests/", "is_input": False, "timeout": None, "reset": False},
        thought="I'll run the tests.",
        timestamp=_T0,
    )
    obs = _obs_event(
        "obs-1", "ExecuteBashObservation",
        {
            "content": [{"type": "text", "text": "5 passed in 0.12s"}],
            "command": "pytest -q tests/",
            "exit_code": 0,
            "error": False,
            "timeout": False,
            "metadata": {"exit_code": 0, "pid": 1, "working_dir": "/repo"},
        },
        action_id="act-1",
        timestamp=_T1,
    )
    return [act, obs]


def _bash_failure_events() -> list[dict]:
    """A failing bash command (exit code 1)."""
    act = _action_event(
        "act-fail", "ExecuteBashAction",
        {"command": "pytest -q tests/", "is_input": False, "timeout": None, "reset": False},
        thought="Running tests.",
        timestamp=_T0,
    )
    obs = _obs_event(
        "obs-fail", "ExecuteBashObservation",
        {
            "content": [{"type": "text", "text": "FAILED tests/test_app.py - 1 failed"}],
            "command": "pytest -q tests/",
            "exit_code": 1,
            "error": True,
            "timeout": False,
            "metadata": {"exit_code": 1, "pid": 2, "working_dir": "/repo"},
        },
        action_id="act-fail",
        timestamp=_T1,
    )
    return [act, obs]


def _file_edit_events(command: str = "str_replace") -> list[dict]:
    """A file editor action with old/new content in observation."""
    act = _action_event(
        "act-edit", "FileEditorAction",
        {
            "command": command,
            "path": "/repo/src/app.py",
            "old_str": 'print("old")',
            "new_str": 'print("new")',
            "file_text": None,
        },
        thought="Patching the file.",
        timestamp=_T2,
    )
    obs = _obs_event(
        "obs-edit", "FileEditorObservation",
        {
            "command": command,
            "path": "/repo/src/app.py",
            "old_content": 'print("old")\n',
            "new_content": 'print("new")\n',
            "prev_exist": True,
            "output": "Edit applied successfully.",
            "error": None,
        },
        action_id="act-edit",
        timestamp=_T3,
    )
    return [act, obs]


def _file_create_events() -> list[dict]:
    """A file create action (old_content is None)."""
    act = _action_event(
        "act-create", "FileEditorAction",
        {
            "command": "create",
            "path": "/repo/src/new_module.py",
            "file_text": "def hello():\n    pass\n",
            "old_str": None,
            "new_str": None,
        },
        timestamp=_T2,
    )
    obs = _obs_event(
        "obs-create", "FileEditorObservation",
        {
            "command": "create",
            "path": "/repo/src/new_module.py",
            "old_content": None,
            "new_content": "def hello():\n    pass\n",
            "prev_exist": False,
            "output": "File created.",
            "error": None,
        },
        action_id="act-create",
        timestamp=_T3,
    )
    return [act, obs]


def _browser_events() -> list[dict]:
    """A browser navigate action."""
    act = _action_event(
        "act-browser", "BrowserNavigateAction",
        {"url": "https://example.com", "new_tab": False},
        thought="Navigating to docs.",
        timestamp=_T1,
    )
    obs = _obs_event(
        "obs-browser", "BrowserObservation",
        {"output": "Page loaded: Example Domain", "error": None, "screenshot_data": None},
        action_id="act-browser",
        timestamp=_T2,
    )
    return [act, obs]


def _mcp_events() -> list[dict]:
    """An MCP tool call action."""
    act = _action_event(
        "act-mcp", "MCPToolAction",
        {"data": {"tool_name": "read_file", "arguments": {"path": "/repo/README.md"}}},
        thought="Reading the README via MCP.",
        timestamp=_T1,
    )
    obs = _obs_event(
        "obs-mcp", "MCPToolObservation",
        {
            "content": [{"type": "text", "text": "# MyProject\n..."}],
            "is_error": False,
            "tool_name": "read_file",
        },
        action_id="act-mcp",
        timestamp=_T2,
    )
    return [act, obs]


def _full_success_conversation() -> list[dict]:
    """A complete successful conversation: message → bash → file edit → finish."""
    events: list[dict] = []
    events.append(_system_event("sys-1", "You are an expert software engineer."))
    events.append(_message_event("msg-1", "user", "Fix the failing assertion.", timestamp=_T0))
    events.extend(_bash_success_events())
    events.extend(_file_edit_events())
    events.append(_finish_action_event("act-finish", "I fixed the assertion. Tests pass.", timestamp=_T4))
    return events


def _full_failure_conversation() -> list[dict]:
    """A conversation that ends with a failed command and no FinishAction."""
    events: list[dict] = []
    events.append(_message_event("msg-1", "user", "Fix the test.", timestamp=_T0))
    events.extend(_bash_failure_events())
    return events


# ---------------------------------------------------------------------------
# Test: conversation_loader
# ---------------------------------------------------------------------------

class TestConversationLoader:
    def test_load_zip_returns_sorted_events(self, tmp_path: Path) -> None:
        events = _bash_success_events()
        meta = {"conversation_id": "conv-1", "title": "Fix tests"}
        zip_bytes = build_conversation_zip(events, meta)
        zip_path = tmp_path / "conv.zip"
        zip_path.write_bytes(zip_bytes)

        conv = load_conversation_zip(zip_path)

        assert conv.meta["conversation_id"] == "conv-1"
        assert len(conv.events) == 2
        assert conv.source_path == str(zip_path)
        # Events should be in filename-sorted order
        assert conv.events[0].get("id") == "act-1"
        assert conv.events[1].get("id") == "obs-1"

    def test_load_dir_returns_sorted_events(self, tmp_path: Path) -> None:
        conv_dir = tmp_path / "conv-xyz"
        conv_dir.mkdir()
        meta = {"conversation_id": "conv-xyz", "model": "claude-sonnet-4-5"}
        (conv_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        events = _bash_success_events()
        for idx, ev in enumerate(events):
            filename = f"2026010{idx}_event_{ev['id']}.json"
            (conv_dir / filename).write_text(json.dumps(ev), encoding="utf-8")

        conv = load_conversation_dir(conv_dir)

        assert conv.meta["conversation_id"] == "conv-xyz"
        assert len(conv.events) == 2

    def test_auto_detect_zip(self, tmp_path: Path) -> None:
        zip_bytes = build_conversation_zip(_bash_success_events(), {})
        zip_path = tmp_path / "conv.zip"
        zip_path.write_bytes(zip_bytes)

        conv = load_conversation(zip_path)
        assert len(conv.events) == 2

    def test_auto_detect_dir(self, tmp_path: Path) -> None:
        conv_dir = tmp_path / "conv-dir"
        conv_dir.mkdir()
        (conv_dir / "meta.json").write_text("{}", encoding="utf-8")
        ev = _bash_success_events()[0]
        (conv_dir / f"20260101_{ev['id']}.json").write_text(json.dumps(ev), encoding="utf-8")

        conv = load_conversation(conv_dir)
        assert len(conv.events) == 1

    def test_missing_zip_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_conversation_zip(tmp_path / "nonexistent.zip")

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_conversation_dir(tmp_path / "nonexistent-dir")

    def test_malformed_event_file_raises(self, tmp_path: Path) -> None:
        conv_dir = tmp_path / "conv-bad"
        conv_dir.mkdir()
        (conv_dir / "meta.json").write_text("{}", encoding="utf-8")
        good_ev = _bash_success_events()[0]
        (conv_dir / "20260101_good.json").write_text(json.dumps(good_ev), encoding="utf-8")
        (conv_dir / "20260102_bad.json").write_text("not json {{", encoding="utf-8")

        with pytest.raises(ValueError, match="Could not parse JSON"):
            load_conversation_dir(conv_dir)

    def test_malformed_zip_event_file_raises(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "conv-bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("meta.json", "{}")
            zf.writestr("event_000001_good.json", json.dumps(_bash_success_events()[0]))
            zf.writestr("event_000002_bad.json", "not json {{")

        with pytest.raises(ValueError, match="Could not parse JSON"):
            load_conversation_zip(zip_path)

    def test_zip_without_meta_gives_empty_meta(self, tmp_path: Path) -> None:
        zip_bytes = build_conversation_zip(_bash_success_events(), None)
        zip_path = tmp_path / "no-meta.zip"
        zip_path.write_bytes(zip_bytes)

        conv = load_conversation_zip(zip_path)
        assert conv.meta == {}
        assert len(conv.events) == 2

    def test_auto_detect_invalid_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be a .zip file or a directory"):
            load_conversation(tmp_path / "unknown.txt")

    def test_dir_uuid_files_sorted_by_timestamp(self, tmp_path: Path) -> None:
        """UUID-named files (real V1 filesystem format) are sorted by event.timestamp."""
        conv_dir = tmp_path / "conv-uuid"
        conv_dir.mkdir()
        ev_late = {
            "id": "late-ev", "timestamp": _T2, "source": "agent",
            "llm_message": {"role": "assistant", "content": "second"},
        }
        ev_early = {
            "id": "early-ev", "timestamp": _T0, "source": "agent",
            "llm_message": {"role": "user", "content": "first"},
        }
        # Write late event with alphabetically earlier filename — sort must ignore filename
        (conv_dir / "aaa-uuid.json").write_text(json.dumps(ev_late), encoding="utf-8")
        (conv_dir / "zzz-uuid.json").write_text(json.dumps(ev_early), encoding="utf-8")

        conv = load_conversation_dir(conv_dir)
        assert len(conv.events) == 2
        assert conv.events[0]["id"] == "early-ev"   # T0 before T2
        assert conv.events[1]["id"] == "late-ev"


# ---------------------------------------------------------------------------
# Test: item_mapper — command execution
# ---------------------------------------------------------------------------

class TestItemMapperCommandExecution:
    def test_bash_action_maps_to_command_execution_event(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_bash_success_events())
        result = map_conversation(conv)

        cmd_events = [e for e in result.events if e.kind == "command_execution"]
        assert len(cmd_events) == 1
        ev = cmd_events[0]
        assert ev.payload["command"] == "pytest -q tests/"
        assert ev.payload["exit_code"] == 0
        assert ev.payload["openhands_action_type"] == "ExecuteBashAction"

    def test_bash_failure_sets_had_command_failure(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_bash_failure_events())
        result = map_conversation(conv)

        cmd_events = [e for e in result.events if e.kind == "command_execution"]
        assert cmd_events[0].payload["exit_code"] == 1
        assert result.outcome.status == "partial"

    def test_terminal_action_also_maps_to_command_execution(self) -> None:
        act = _action_event(
            "act-term", "TerminalAction",
            {"command": "ls /repo", "is_input": False, "timeout": None, "reset": False},
        )
        obs = _obs_event(
            "obs-term", "TerminalObservation",
            {
                "content": [{"type": "text", "text": "src/\ntests/"}],
                "command": "ls /repo",
                "exit_code": 0,
                "is_error": False,
                "timeout": False,
                "metadata": {"exit_code": 0},
            },
            action_id="act-term",
        )
        conv = OpenHandsConversation(meta={}, events=[act, obs])
        result = map_conversation(conv)

        cmd_events = [e for e in result.events if e.kind == "command_execution"]
        assert len(cmd_events) == 1
        assert cmd_events[0].payload["openhands_action_type"] == "TerminalAction"

    def test_test_command_populates_test_trace(self) -> None:
        act = _action_event(
            "act-pytest", "ExecuteBashAction",
            {"command": "pytest -q tests/test_app.py", "is_input": False, "timeout": None, "reset": False},
        )
        obs = _obs_event(
            "obs-pytest", "ExecuteBashObservation",
            {
                "content": [{"type": "text", "text": "2 passed in 0.05s"}],
                "exit_code": 0,
                "error": False,
                "timeout": False,
            },
            action_id="act-pytest",
        )
        conv = OpenHandsConversation(meta={}, events=[act, obs])
        result = map_conversation(conv)

        assert result.test_trace is not None
        assert len(result.test_trace) == 1
        assert result.test_trace[0]["command"] == "pytest -q tests/test_app.py"
        assert result.test_trace[0]["passed"] is True
        assert result.stdout_excerpt == "2 passed in 0.05s"

    def test_timed_out_command_counts_as_failure(self) -> None:
        act = _action_event(
            "act-timeout", "ExecuteBashAction",
            {"command": "sleep 100", "is_input": False, "timeout": 5, "reset": False},
        )
        obs = _obs_event(
            "obs-timeout", "ExecuteBashObservation",
            {
                "content": [{"type": "text", "text": "Process timed out."}],
                "exit_code": None,
                "error": True,
                "timeout": True,
            },
            action_id="act-timeout",
        )
        conv = OpenHandsConversation(meta={}, events=[act, obs])
        result = map_conversation(conv)

        cmd_events = [e for e in result.events if e.kind == "command_execution"]
        assert cmd_events[0].payload["timed_out"] is True
        assert result.outcome.status == "partial"

    def test_command_without_paired_observation_still_maps(self) -> None:
        act = _action_event(
            "act-solo", "ExecuteBashAction",
            {"command": "echo hello", "is_input": False, "timeout": None, "reset": False},
        )
        conv = OpenHandsConversation(meta={}, events=[act])
        result = map_conversation(conv)

        cmd_events = [e for e in result.events if e.kind == "command_execution"]
        assert len(cmd_events) == 1
        assert cmd_events[0].payload["exit_code"] is None

    def test_tool_trace_contains_all_command_payloads(self) -> None:
        events = []
        for i in range(3):
            act = _action_event(
                f"act-{i}", "ExecuteBashAction",
                {"command": f"echo {i}", "is_input": False, "timeout": None, "reset": False},
                timestamp=f"2026-01-01T00:00:0{i}.000Z",
            )
            events.append(act)
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.tool_trace is not None
        assert len(result.tool_trace) == 3


# ---------------------------------------------------------------------------
# Test: item_mapper — file changes
# ---------------------------------------------------------------------------

class TestItemMapperFileChanges:
    def test_str_replace_maps_to_file_change_event(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_file_edit_events("str_replace"))
        result = map_conversation(conv)

        file_events = [e for e in result.events if e.kind == "file_change"]
        assert len(file_events) == 1
        ev = file_events[0]
        assert ev.payload["path"] == "repo/src/app.py"
        assert ev.payload["operation"] == "str_replace"

    def test_file_view_action_is_not_emitted_as_file_change(self) -> None:
        act = _action_event(
            "act-view", "FileEditorAction",
            {"command": "view", "path": "/repo/src/app.py"},
        )
        obs = _obs_event(
            "obs-view", "FileEditorObservation",
            {"command": "view", "path": "/repo/src/app.py", "old_content": None, "new_content": None, "output": "..."},
            action_id="act-view",
        )
        conv = OpenHandsConversation(meta={}, events=[act, obs])
        result = map_conversation(conv)

        assert not any(e.kind == "file_change" for e in result.events)
        assert result.outcome.files_touched == []

    def test_create_action_makes_creation_diff(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_file_create_events())
        result = map_conversation(conv)

        file_events = [e for e in result.events if e.kind == "file_change"]
        assert len(file_events) == 1
        assert file_events[0].payload["path"] == "repo/src/new_module.py"
        # Diff is present (created from None → new content)
        assert result.patch_text is not None
        assert "def hello" in result.patch_text
        assert result.patch_hash is not None
        assert result.patch_hash.startswith("sha256:")

    def test_multiple_file_edits_aggregate_into_files_touched(self) -> None:
        events = []
        for i, fname in enumerate(["src/a.py", "src/b.py"]):
            act = _action_event(
                f"act-edit-{i}", "FileEditorAction",
                {"command": "str_replace", "path": f"/repo/{fname}"},
                timestamp=f"2026-01-01T00:00:0{i}.000Z",
            )
            obs = _obs_event(
                f"obs-edit-{i}", "FileEditorObservation",
                {
                    "command": "str_replace",
                    "path": f"/repo/{fname}",
                    "old_content": "old\n",
                    "new_content": "new\n",
                    "output": "ok",
                    "error": None,
                },
                action_id=f"act-edit-{i}",
                timestamp=f"2026-01-01T00:00:0{i + 1}.000Z",
            )
            events.extend([act, obs])
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert sorted(result.outcome.files_touched) == ["repo/src/a.py", "repo/src/b.py"]
        assert result.patch_text is not None
        # Both file diffs combined
        assert "a.py" in result.patch_text
        assert "b.py" in result.patch_text

    def test_file_change_without_observation_has_no_patch_hash(self) -> None:
        act = _action_event(
            "act-edit-solo", "FileEditorAction",
            {"command": "str_replace", "path": "/repo/src/app.py"},
        )
        conv = OpenHandsConversation(meta={}, events=[act])
        result = map_conversation(conv)

        file_events = [e for e in result.events if e.kind == "file_change"]
        assert len(file_events) == 1
        assert file_events[0].payload["patch_hash"] is None
        assert result.patch_text is None

    def test_str_replace_editor_action_alias_maps_correctly(self) -> None:
        act = _action_event(
            "act-sse", "StrReplaceEditorAction",
            {"command": "str_replace", "path": "/repo/src/app.py"},
        )
        obs = _obs_event(
            "obs-sse", "FileEditorObservation",
            {"command": "str_replace", "path": "/repo/src/app.py",
             "old_content": "x\n", "new_content": "y\n", "output": "ok", "error": None},
            action_id="act-sse",
        )
        conv = OpenHandsConversation(meta={}, events=[act, obs])
        result = map_conversation(conv)

        file_events = [e for e in result.events if e.kind == "file_change"]
        assert len(file_events) == 1
        assert result.patch_text is not None


# ---------------------------------------------------------------------------
# Test: item_mapper — browser / non-code actions
# ---------------------------------------------------------------------------

class TestItemMapperBrowserActions:
    def test_browser_navigate_maps_to_agent_message(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_browser_events())
        result = map_conversation(conv)

        browser_events = [e for e in result.events if e.kind == "agent_message"
                          and e.payload.get("source") == "openhands_browser_action"]
        assert len(browser_events) == 1
        ev = browser_events[0]
        assert ev.payload["openhands_action_type"] == "BrowserNavigateAction"
        assert ev.payload["url"] == "https://example.com"
        assert ev.payload["obs_output"] == "Page loaded: Example Domain"

    def test_browser_click_maps_to_agent_message(self) -> None:
        act = _action_event("act-click", "BrowserClickAction", {"index": 3, "new_tab": False})
        conv = OpenHandsConversation(meta={}, events=[act])
        result = map_conversation(conv)

        browser_events = [e for e in result.events if e.kind == "agent_message"
                          and e.payload.get("openhands_action_type") == "BrowserClickAction"]
        assert len(browser_events) == 1
        assert browser_events[0].payload["index"] == 3

    def test_browser_actions_never_create_file_change_events(self) -> None:
        events = _browser_events()
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert not any(e.kind == "file_change" for e in result.events)
        assert result.outcome.files_touched == []


# ---------------------------------------------------------------------------
# Test: item_mapper — MCP tool calls
# ---------------------------------------------------------------------------

class TestItemMapperMCP:
    def test_mcp_tool_action_maps_to_mcp_tool_call(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_mcp_events())
        result = map_conversation(conv)

        mcp_events = [e for e in result.events if e.kind == "mcp_tool_call"]
        assert len(mcp_events) == 1
        ev = mcp_events[0]
        assert ev.payload["tool_name"] == "read_file"
        assert ev.payload["is_error"] is False
        assert ev.payload["data"]["tool_name"] == "read_file"

    def test_mcp_error_observation_sets_is_error(self) -> None:
        act = _action_event(
            "act-mcp-err", "MCPToolAction",
            {"data": {"tool_name": "write_file", "arguments": {}}},
        )
        obs = _obs_event(
            "obs-mcp-err", "MCPToolObservation",
            {"content": [], "is_error": True, "tool_name": "write_file"},
            action_id="act-mcp-err",
        )
        conv = OpenHandsConversation(meta={}, events=[act, obs])
        result = map_conversation(conv)

        mcp_events = [e for e in result.events if e.kind == "mcp_tool_call"]
        assert mcp_events[0].payload["is_error"] is True


# ---------------------------------------------------------------------------
# Test: item_mapper — message and system events
# ---------------------------------------------------------------------------

class TestItemMapperMessages:
    def test_message_event_maps_to_agent_message(self) -> None:
        ev = _message_event("msg-1", "assistant", "I will fix this.", activated_microagents=["fix"])
        conv = OpenHandsConversation(meta={}, events=[ev])
        result = map_conversation(conv)

        msg_events = [e for e in result.events if e.kind == "agent_message"
                      and e.payload.get("source") == "openhands_message"]
        assert len(msg_events) == 1
        assert msg_events[0].payload["role"] == "assistant"
        assert msg_events[0].payload["text"] == "I will fix this."
        assert msg_events[0].payload["activated_microagents"] == ["fix"]

    def test_user_message_event_is_captured(self) -> None:
        ev = _message_event("msg-u", "user", "Please fix the test.", timestamp=_T0)
        conv = OpenHandsConversation(meta={}, events=[ev])
        result = map_conversation(conv)

        msg_events = [e for e in result.events if e.kind == "agent_message"
                      and e.payload.get("role") == "user"]
        assert len(msg_events) == 1

    def test_system_event_maps_to_agent_message(self) -> None:
        ev = _system_event("sys-1", "You are a helpful engineer.")
        conv = OpenHandsConversation(meta={}, events=[ev])
        result = map_conversation(conv)

        sys_events = [e for e in result.events if e.kind == "agent_message"
                      and e.payload.get("source") == "openhands_system"]
        assert len(sys_events) == 1
        assert "helpful engineer" in sys_events[0].payload["system_prompt"]

    def test_hook_event_maps_to_agent_message(self) -> None:
        hook_ev = {
            "id": "hook-1",
            "timestamp": _T0,
            "source": "hook",
            "kind": "HookExecutionEvent",
            "hook_event_type": "PreToolUse",
            "hook_command": "pre_action",
            "success": True,
            "blocked": False,
            "exit_code": 0,
        }
        conv = OpenHandsConversation(meta={}, events=[hook_ev])
        result = map_conversation(conv)

        hook_events = [e for e in result.events if e.kind == "agent_message"
                       and e.payload.get("source") == "openhands_hook"]
        assert len(hook_events) == 1
        assert hook_events[0].payload["hook_command"] == "pre_action"
        assert hook_events[0].payload["hook_event_type"] == "PreToolUse"
        assert hook_events[0].payload["success"] is True
        assert hook_events[0].payload["blocked"] is False

    def test_hook_payload_captures_tool_name_stdout_stderr_error(self) -> None:
        hook_ev = {
            "id": "hook-2",
            "timestamp": _T0,
            "source": "hook",
            "kind": "HookExecutionEvent",
            "hook_event_type": "PostToolUse",
            "hook_command": "post_script.sh",
            "success": False,
            "blocked": True,
            "exit_code": 1,
            "reason": "Policy violation",
            "tool_name": "bash",
            "action_id": "act-123",
            "message_id": "msg-123",
            "stdout": "output line",
            "stderr": "error line",
            "error": "Execution failed",
            "additional_context": "extra info",
            "hook_input": {"tool": "bash", "command": "pytest"},
        }
        conv = OpenHandsConversation(meta={}, events=[hook_ev])
        result = map_conversation(conv)

        hook_events = [e for e in result.events if e.payload.get("source") == "openhands_hook"]
        assert len(hook_events) == 1
        p = hook_events[0].payload
        assert p["tool_name"] == "bash"
        assert p["action_id"] == "act-123"
        assert p["message_id"] == "msg-123"
        assert p["stdout"] == "output line"
        assert p["stderr"] == "error line"
        assert p["error"] == "Execution failed"
        assert p["additional_context"] == "extra info"
        assert p["hook_input"] == {"tool": "bash", "command": "pytest"}
        assert p["blocked"] is True
        assert p["reason"] == "Policy violation"


# ---------------------------------------------------------------------------
# Test: item_mapper — FinishAction and final response
# ---------------------------------------------------------------------------

class TestItemMapperFinishAction:
    def test_finish_action_sets_final_response(self) -> None:
        events = [_finish_action_event("act-fin", "Task complete. Tests pass.")]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.outcome.final_response == "Task complete. Tests pass."

    def test_finish_action_emits_agent_message_event(self) -> None:
        events = [_finish_action_event("act-fin", "Done.")]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        finish_events = [e for e in result.events if e.kind == "agent_message"
                         and e.payload.get("source") == "openhands_finish"]
        assert len(finish_events) == 1
        assert finish_events[0].payload["message"] == "Done."

    def test_finish_action_gives_completed_status(self) -> None:
        events = [_finish_action_event("act-fin", "All done.")]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.outcome.status == "completed"

    def test_finish_action_with_prior_command_failure_gives_partial(self) -> None:
        events = [*_bash_failure_events(), _finish_action_event("act-fin", "Done despite errors.")]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.outcome.status == "partial"
        assert result.outcome.final_response == "Done despite errors."


# ---------------------------------------------------------------------------
# Test: item_mapper — failed run
# ---------------------------------------------------------------------------

class TestItemMapperFailedRun:
    def test_no_events_gives_failed_outcome(self) -> None:
        conv = OpenHandsConversation(meta={}, events=[])
        result = map_conversation(conv)

        assert result.outcome.status == "failed"
        assert result.outcome.escalation_reason is not None

    def test_command_failure_without_finish_gives_partial(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_bash_failure_events())
        result = map_conversation(conv)

        assert result.outcome.status == "partial"
        assert result.outcome.files_touched == []
        assert result.patch_text is None

    def test_agent_error_event_gives_partial_when_events_present(self) -> None:
        events = [
            *_bash_success_events(),
            _error_event("err-1", "LLM returned invalid JSON."),
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.outcome.status == "partial"
        error_events = [e for e in result.events if e.kind == "agent_message"
                        and e.payload.get("source") == "openhands_agent_error"]
        assert len(error_events) == 1

    def test_agent_error_only_gives_failed_outcome(self) -> None:
        events = [_error_event("err-1", "Fatal error.")]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        # Only error event was emitted; no actionable events → failed
        assert result.outcome.status in {"failed", "partial"}

    def test_conversation_state_error_sets_fatal_flag(self) -> None:
        events = [
            *_bash_success_events(),
            _state_event("state-1", "error", timestamp=_T3),
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        # Had bash success events, so outcome is partial (not failed)
        assert result.outcome.status == "partial"

    def test_failure_stderr_excerpt_captures_last_failure(self) -> None:
        events = [
            *_bash_success_events(),
            *_bash_failure_events(),
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.stderr_excerpt is not None
        assert "FAILED" in result.stderr_excerpt

    def test_conversation_error_event_sets_fatal_flag(self) -> None:
        events = [
            *_bash_success_events(),
            {
                "id": "err-conv",
                "timestamp": _T3,
                "source": "environment",
                "kind": "ConversationErrorEvent",
                "code": "AGENT_ERROR",
                "detail": "Agent ran out of iterations.",
            },
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.outcome.status == "partial"
        error_events = [e for e in result.events if e.kind == "agent_message"
                        and e.payload.get("source") == "openhands_error"]
        assert len(error_events) == 1
        assert error_events[0].payload["error_kind"] == "ConversationErrorEvent"

    def test_server_error_event_sets_fatal_flag(self) -> None:
        events = [
            {
                "id": "srv-err",
                "timestamp": _T0,
                "source": "environment",
                "kind": "ServerErrorEvent",
                "code": "INTERNAL",
                "detail": "Runtime crashed.",
            },
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        # Error event emitted; had_fatal_error set; with at least one event → partial
        assert result.outcome.status in {"failed", "partial"}
        error_events = [e for e in result.events if e.payload.get("source") == "openhands_error"]
        assert len(error_events) == 1
        assert error_events[0].payload["error_kind"] == "ServerErrorEvent"


# ---------------------------------------------------------------------------
# Test: item_mapper — ConversationStateEvent "finished"
# ---------------------------------------------------------------------------

class TestItemMapperConversationState:
    def test_state_finished_with_no_finish_action_gives_completed(self) -> None:
        events = [
            *_bash_success_events(),
            _state_event("state-fin", "finished", timestamp=_T3),
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.outcome.status == "completed"

    def test_state_event_itself_not_emitted_as_cl_event(self) -> None:
        events = [_state_event("state-1", "running")]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        # No CL events emitted for state-only events
        assert len(result.events) == 0
        assert result.outcome.status == "partial"
        assert result.outcome.escalation_reason == "conversation did not reach a terminal state"

    def test_running_state_after_message_is_partial_not_completed(self) -> None:
        events = [
            _message_event("msg-1", "user", "Please fix the test.", timestamp=_T0),
            _state_event("state-1", "running", timestamp=_T1),
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.outcome.status == "partial"
        assert result.outcome.escalation_reason == "conversation did not reach a terminal state"

    def test_stats_event_extracts_token_counts(self) -> None:
        stats_value = {
            "usage_to_metrics": {
                "agent": {
                    "accumulated_token_usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                    }
                }
            }
        }
        stats_ev = {
            "id": "stats-1",
            "timestamp": _T1,
            "source": "environment",
            "kind": "ConversationStateUpdateEvent",
            "key": "stats",
            "value": stats_value,
        }
        conv = OpenHandsConversation(meta={}, events=[stats_ev])
        result = map_conversation(conv)

        assert result.cost_tokens_prompt == 1000
        assert result.cost_tokens_completion == 500

    def test_repeated_stats_events_keep_latest_accumulated_snapshot(self) -> None:
        first_stats = {
            "usage_to_metrics": {
                "agent": {
                    "accumulated_token_usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                    }
                }
            }
        }
        latest_stats = {
            "usage_to_metrics": {
                "agent": {
                    "accumulated_token_usage": {
                        "prompt_tokens": 1500,
                        "completion_tokens": 700,
                    }
                }
            }
        }
        events = [
            {
                "id": "stats-1",
                "timestamp": _T1,
                "source": "environment",
                "kind": "ConversationStateUpdateEvent",
                "key": "stats",
                "value": first_stats,
            },
            {
                "id": "stats-2",
                "timestamp": _T2,
                "source": "environment",
                "kind": "ConversationStateUpdateEvent",
                "key": "stats",
                "value": latest_stats,
            },
        ]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        assert result.cost_tokens_prompt == 1500
        assert result.cost_tokens_completion == 700

    def test_full_state_event_extracts_execution_status_and_tokens(self) -> None:
        full_state_value = {
            "execution_status": "finished",
            "stats": {
                "usage_to_metrics": {
                    "agent": {
                        "accumulated_token_usage": {
                            "prompt_tokens": 2000,
                            "completion_tokens": 800,
                        }
                    }
                }
            },
        }
        full_state_ev = {
            "id": "fs-1",
            "timestamp": _T2,
            "source": "environment",
            "kind": "ConversationStateUpdateEvent",
            "key": "full_state",
            "value": full_state_value,
        }
        conv = OpenHandsConversation(meta={}, events=[full_state_ev])
        result = map_conversation(conv)

        assert result.outcome.status == "completed"   # execution_status == "finished"
        assert result.cost_tokens_prompt == 2000
        assert result.cost_tokens_completion == 800

    def test_full_state_error_status_sets_fatal_flag(self) -> None:
        full_state_ev = {
            "id": "fs-err",
            "timestamp": _T2,
            "source": "environment",
            "kind": "ConversationStateUpdateEvent",
            "key": "full_state",
            "value": {"execution_status": "error"},
        }
        conv = OpenHandsConversation(meta={}, events=[*_bash_success_events(), full_state_ev])
        result = map_conversation(conv)

        assert result.outcome.status == "partial"


# ---------------------------------------------------------------------------
# Test: full conversation integration
# ---------------------------------------------------------------------------

class TestFullConversation:
    def test_meta_metrics_are_used_as_token_fallback(self) -> None:
        conv = OpenHandsConversation(
            meta={
                "conversation_id": "metric-conv",
                "metrics": {
                    "accumulated_token_usage": {
                        "prompt_tokens": 321,
                        "completion_tokens": 123,
                    }
                },
            },
            events=_bash_success_events(),
        )
        result = map_conversation(conv)

        assert result.cost_tokens_prompt == 321
        assert result.cost_tokens_completion == 123

    def test_llm_model_meta_key_is_used_for_base_model_id(self) -> None:
        conv = OpenHandsConversation(
            meta={"conversation_id": "m-conv", "llm_model": "claude-haiku-4-5"},
            events=_bash_success_events(),
        )
        result = map_conversation(conv)

        assert result.base_model_id == "claude-haiku-4-5"

    def test_llm_model_takes_precedence_over_model_key(self) -> None:
        conv = OpenHandsConversation(
            meta={"llm_model": "claude-haiku-4-5", "model": "other-model"},
            events=_bash_success_events(),
        )
        result = map_conversation(conv)

        assert result.base_model_id == "claude-haiku-4-5"

    def test_full_success_conversation_maps_all_event_kinds(self) -> None:
        conv = OpenHandsConversation(
            meta={"conversation_id": "conv-full", "model": "claude-opus-4-7"},
            events=_full_success_conversation(),
        )
        result = map_conversation(conv)

        kinds = {e.kind for e in result.events}
        assert "command_execution" in kinds
        assert "file_change" in kinds
        assert "agent_message" in kinds

        assert result.outcome.status == "completed"
        assert result.outcome.final_response == "I fixed the assertion. Tests pass."
        assert "repo/src/app.py" in result.outcome.files_touched
        assert result.patch_text is not None
        assert result.patch_hash is not None
        assert result.base_model_id == "claude-opus-4-7"

    def test_full_failure_conversation_is_partial(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_full_failure_conversation())
        result = map_conversation(conv)

        assert result.outcome.status == "partial"
        assert result.outcome.final_response is None
        assert result.outcome.files_touched == []

    def test_think_action_maps_to_agent_message(self) -> None:
        act = _action_event("act-think", "ThinkAction", {"thought": "Analyzing the codebase."})
        conv = OpenHandsConversation(meta={}, events=[act])
        result = map_conversation(conv)

        think_events = [e for e in result.events if e.kind == "agent_message"
                        and e.payload.get("source") == "openhands_think"]
        assert len(think_events) == 1
        assert "Analyzing" in think_events[0].payload["thought"]

    def test_glob_action_maps_to_agent_message(self) -> None:
        act = _action_event("act-glob", "GlobAction", {"pattern": "**/*.py", "path": "/repo"})
        conv = OpenHandsConversation(meta={}, events=[act])
        result = map_conversation(conv)

        glob_events = [e for e in result.events if e.kind == "agent_message"
                       and e.payload.get("openhands_action_type") == "GlobAction"]
        assert len(glob_events) == 1
        assert glob_events[0].payload["pattern"] == "**/*.py"

    def test_task_tracker_action_maps_to_agent_message(self) -> None:
        act = _action_event("act-task", "TaskTrackerAction",
                            {"command": "view", "task_list": []})
        conv = OpenHandsConversation(meta={}, events=[act])
        result = map_conversation(conv)

        task_events = [e for e in result.events if e.kind == "agent_message"
                       and e.payload.get("openhands_action_type") == "TaskTrackerAction"]
        assert len(task_events) == 1

    def test_unknown_action_type_maps_to_agent_message(self) -> None:
        act = _action_event("act-unk", "FutureAction", {"param": "value"})
        conv = OpenHandsConversation(meta={}, events=[act])
        result = map_conversation(conv)

        unk_events = [e for e in result.events if e.kind == "agent_message"
                      and e.payload.get("openhands_action_type") == "FutureAction"]
        assert len(unk_events) == 1

    def test_condensation_event_is_silently_skipped(self) -> None:
        cond_ev = {
            "id": "cond-1",
            "timestamp": _T0,
            "source": "agent",
            "forgotten_event_ids": ["evt-1", "evt-2"],
            "summary": "Summary of events.",
        }
        conv = OpenHandsConversation(meta={}, events=[cond_ev])
        result = map_conversation(conv)

        assert len(result.events) == 0

    def test_observation_only_events_are_not_double_counted(self) -> None:
        """Observations that appear as standalone events should not produce
        duplicate CL events (they are consumed via the obs_index)."""
        events = _bash_success_events()  # [action, observation]
        conv = OpenHandsConversation(meta={}, events=events)
        result = map_conversation(conv)

        cmd_events = [e for e in result.events if e.kind == "command_execution"]
        assert len(cmd_events) == 1  # exactly one, not two


# ---------------------------------------------------------------------------
# Test: conversation_to_episode and import_conversation
# ---------------------------------------------------------------------------

class TestConversationToEpisode:
    def test_episode_fields_populated_from_meta(self) -> None:
        conv = OpenHandsConversation(
            meta={
                "conversation_id": "conv-abc",
                "title": "Fix assertion error",
                "created_at": "2026-03-01T12:00:00Z",
                "ended_at": "2026-03-01T12:05:00Z",
                "model": "gpt-4o",
            },
            events=_full_success_conversation(),
            source_path="/exports/conv-abc.zip",
        )

        episode = conversation_to_episode(
            conv, task_id="task-001", task_domain="python", mode="baseline"
        )

        assert episode.agent_surface == "openhands"
        assert episode.task_id == "task-001"
        assert episode.task_domain == "python"
        assert episode.mode == "baseline"
        assert episode.thread_id == "conv-abc"
        assert episode.task_description == "Fix assertion error"
        assert episode.base_model_id == "gpt-4o"
        assert episode.reward is None
        assert episode.started_at == _ts("2026-03-01T12:00:00Z")
        assert episode.ended_at == _ts("2026-03-01T12:05:00Z")

    def test_episode_outcome_reflects_success(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_full_success_conversation())
        episode = conversation_to_episode(conv, task_domain="python")

        assert episode.outcome.status == "completed"
        assert episode.outcome.final_response is not None
        assert "repo/src/app.py" in episode.outcome.files_touched

    def test_episode_patch_text_and_hash_set_when_diffs_present(self) -> None:
        conv = OpenHandsConversation(meta={}, events=_file_edit_events())
        episode = conversation_to_episode(conv, task_domain="python")

        assert episode.patch_text is not None
        assert episode.patch_hash is not None
        assert episode.patch_hash.startswith("sha256:")

    def test_episode_id_is_stable_across_imports(self) -> None:
        conv = OpenHandsConversation(
            meta={"conversation_id": "stable-conv"},
            events=_bash_success_events(),
        )
        ep1 = conversation_to_episode(conv, mode="baseline")
        ep2 = conversation_to_episode(conv, mode="baseline")

        assert ep1.episode_id == ep2.episode_id

    def test_different_modes_produce_different_episode_ids(self) -> None:
        conv = OpenHandsConversation(
            meta={"conversation_id": "mode-conv"},
            events=_bash_success_events(),
        )
        ep_base = conversation_to_episode(conv, mode="baseline")
        ep_intg = conversation_to_episode(conv, mode="integrated")

        assert ep_base.episode_id != ep_intg.episode_id

    def test_invalid_mode_raises(self) -> None:
        conv = OpenHandsConversation(meta={}, events=[])
        with pytest.raises(ValueError, match="mode"):
            conversation_to_episode(conv, mode="unknown")  # type: ignore

    def test_import_conversation_from_zip(self, tmp_path: Path) -> None:
        meta = {"conversation_id": "zip-conv", "model": "test-model"}
        events = _full_success_conversation()
        zip_bytes = build_conversation_zip(events, meta)
        zip_path = tmp_path / "conv.zip"
        zip_path.write_bytes(zip_bytes)

        episode = import_conversation(zip_path, task_id="zip-task", task_domain="python")

        assert episode.task_id == "zip-task"
        assert episode.base_model_id == "test-model"
        assert episode.outcome.status == "completed"

    def test_import_conversation_from_dir(self, tmp_path: Path) -> None:
        conv_dir = tmp_path / "conv-dir"
        conv_dir.mkdir()
        meta = {"conversation_id": "dir-conv"}
        (conv_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        events = _bash_success_events()
        for idx, ev in enumerate(events):
            (conv_dir / f"20260101_{idx:04d}_{ev['id']}.json").write_text(
                json.dumps(ev), encoding="utf-8"
            )

        episode = import_conversation(conv_dir, task_domain="python")
        assert episode.thread_id == "dir-conv"


# ---------------------------------------------------------------------------
# Test: recorder round-trip
# ---------------------------------------------------------------------------

class TestRecorderRoundtrip:
    def test_append_and_reload_preserves_episode_fields(self, tmp_path: Path) -> None:
        events = _full_success_conversation()
        meta = {"conversation_id": "rt-conv", "model": "claude-sonnet-4-6",
                "title": "Recorder round-trip test"}
        zip_bytes = build_conversation_zip(events, meta)
        zip_path = tmp_path / "rt-conv.zip"
        zip_path.write_bytes(zip_bytes)

        episodes_path = tmp_path / "episodes.jsonl"
        appended = append_conversation_episode(
            zip_path, episodes_path, task_id="rt-task", task_domain="python"
        )

        loaded = EpisodeRecorder(episodes_path).load_all()
        assert len(loaded) == 1
        ep = loaded[0]
        assert ep.episode_id == appended.episode_id
        assert ep.task_id == "rt-task"
        assert ep.agent_surface == "openhands"
        assert ep.outcome.status == "completed"
        assert ep.patch_text == appended.patch_text
        assert ep.patch_hash == appended.patch_hash
        assert ep.base_model_id == "claude-sonnet-4-6"
        assert ep.reward is None

    def test_multiple_episodes_append_independently(self, tmp_path: Path) -> None:
        episodes_path = tmp_path / "episodes.jsonl"

        for i, (conv_events, task_id) in enumerate(
            [(_full_success_conversation(), "task-A"), (_full_failure_conversation(), "task-B")]
        ):
            meta = {"conversation_id": f"conv-{i}"}
            zip_bytes = build_conversation_zip(conv_events, meta)
            zip_path = tmp_path / f"conv-{i}.zip"
            zip_path.write_bytes(zip_bytes)
            append_conversation_episode(zip_path, episodes_path, task_id=task_id)

        loaded = EpisodeRecorder(episodes_path).load_all()
        assert len(loaded) == 2
        assert {ep.task_id for ep in loaded} == {"task-A", "task-B"}

    def test_episode_events_serialize_and_deserialize_correctly(self, tmp_path: Path) -> None:
        events = _full_success_conversation()
        conv = OpenHandsConversation(meta={"conversation_id": "ser-conv"}, events=events)
        original = conversation_to_episode(conv, task_domain="python")

        episodes_path = tmp_path / "episodes.jsonl"
        EpisodeRecorder(episodes_path).append(original)
        loaded = EpisodeRecorder(episodes_path).load_all()

        assert len(loaded) == 1
        restored = loaded[0]
        assert len(restored.events) == len(original.events)
        for orig_ev, rest_ev in zip(original.events, restored.events):
            assert orig_ev.kind == rest_ev.kind
            assert orig_ev.payload == rest_ev.payload


# ---------------------------------------------------------------------------
# Test: context builder
# ---------------------------------------------------------------------------

class TestContextBuilder:
    def test_baseline_mode_injects_nothing(self, tmp_path: Path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "PROGRAM.md").write_text("## Prior context", encoding="utf-8")
        (artifacts / "SKILLS.md").write_text("## Skills", encoding="utf-8")

        builder = ContextBuilder(artifacts)
        ctx = builder.build("Fix the bug.", mode="baseline", cwd=str(tmp_path))

        assert ctx.injected_artifacts == []
        assert ctx.skill_dir is None
        assert not (tmp_path / ".openhands").exists()

    def test_integrated_mode_writes_to_skills_dir(self, tmp_path: Path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "PROGRAM.md").write_text("## Program context", encoding="utf-8")
        (artifacts / "SKILLS.md").write_text("## Skills list", encoding="utf-8")
        repo = tmp_path / "repo"
        repo.mkdir()

        builder = ContextBuilder(artifacts)
        ctx = builder.build("Fix the bug.", mode="integrated", cwd=str(repo))

        assert "cl-program.md" in ctx.injected_artifacts
        assert "cl-skills.md" in ctx.injected_artifacts
        assert ctx.skill_dir is not None
        skills_dir = Path(ctx.skill_dir)
        assert (skills_dir / "cl-program.md").read_text() == "## Program context"
        assert (skills_dir / "cl-skills.md").read_text() == "## Skills list"

    def test_integrated_mode_preserves_existing_openhands_skills(
        self, tmp_path: Path
    ) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "PROGRAM.md").write_text("## CL program", encoding="utf-8")
        (artifacts / "SKILLS.md").write_text("## CL skills", encoding="utf-8")
        repo = tmp_path / "repo"
        skills_dir = repo / ".openhands" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "PROGRAM.md").write_text("## User program", encoding="utf-8")
        (skills_dir / "SKILLS.md").write_text("## User skills", encoding="utf-8")

        builder = ContextBuilder(artifacts)
        ctx = builder.build("Fix the bug.", mode="integrated", cwd=str(repo))

        assert ctx.skill_dir == str(skills_dir)
        assert (skills_dir / "PROGRAM.md").read_text() == "## User program"
        assert (skills_dir / "SKILLS.md").read_text() == "## User skills"
        assert (skills_dir / "cl-program.md").read_text() == "## CL program"
        assert (skills_dir / "cl-skills.md").read_text() == "## CL skills"

    def test_integrated_mode_with_no_artifacts_dir_injects_nothing(self, tmp_path: Path) -> None:
        builder = ContextBuilder(None)
        ctx = builder.build("Fix the bug.", mode="integrated", cwd=str(tmp_path))

        assert ctx.injected_artifacts == []

    def test_integrated_mode_without_cwd_injects_nothing(self) -> None:
        builder = ContextBuilder(None)
        ctx = builder.build("Fix the bug.", mode="integrated", cwd=None)

        assert ctx.skill_dir is None
        assert ctx.injected_artifacts == []

    def test_inject_dir_overrides_cwd(self, tmp_path: Path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "PROGRAM.md").write_text("content", encoding="utf-8")

        cwd = tmp_path / "cwd"
        cwd.mkdir()
        inject_dir = tmp_path / "inject"
        inject_dir.mkdir()

        builder = ContextBuilder(artifacts)
        ctx = builder.build("task", mode="integrated", cwd=str(cwd), inject_dir=str(inject_dir))

        assert ctx.skill_dir is not None
        assert str(inject_dir) in ctx.skill_dir
        assert not (cwd / ".openhands").exists()

    def test_run_context_dataclass_fields(self) -> None:
        ctx = OpenHandsRunContext(
            task_prompt="Fix it.", mode="baseline", cwd="/repo"
        )
        assert ctx.injected_artifacts == []
        assert ctx.skill_dir is None


# ---------------------------------------------------------------------------
# Test: OpenHandsImporter class
# ---------------------------------------------------------------------------

class TestOpenHandsImporter:
    def test_importer_import_conversation_appends_to_recorder(self, tmp_path: Path) -> None:
        zip_bytes = build_conversation_zip(_full_success_conversation(),
                                           {"conversation_id": "imp-1"})
        zip_path = tmp_path / "imp-1.zip"
        zip_path.write_bytes(zip_bytes)
        episodes_path = tmp_path / "ep.jsonl"

        importer = OpenHandsImporter(episodes_path)
        episode = importer.import_conversation(zip_path, task_id="t1", task_domain="python")

        assert episode.task_id == "t1"
        loaded = EpisodeRecorder(episodes_path).load_all()
        assert len(loaded) == 1
        assert loaded[0].episode_id == episode.episode_id

    def test_importer_import_conversations_batch(self, tmp_path: Path) -> None:
        paths = []
        for i in range(3):
            ev_list = _bash_success_events() if i % 2 == 0 else _bash_failure_events()
            zb = build_conversation_zip(ev_list, {"conversation_id": f"batch-{i}"})
            zp = tmp_path / f"batch-{i}.zip"
            zp.write_bytes(zb)
            paths.append(zp)

        episodes_path = tmp_path / "batch.jsonl"
        importer = OpenHandsImporter(episodes_path)
        episodes = importer.import_conversations(paths)

        assert len(episodes) == 3
        assert len(EpisodeRecorder(episodes_path).load_all()) == 3


# ---------------------------------------------------------------------------
# Test: OpenHandsLiveRunner is deferred
# ---------------------------------------------------------------------------

class TestOpenHandsLiveRunner:
    def test_live_runner_raises_without_client(self) -> None:
        with pytest.raises(RuntimeError, match="api_client"):
            OpenHandsLiveRunner()

    def test_live_runner_run_not_implemented(self) -> None:
        fake_client = object()
        runner = OpenHandsLiveRunner(api_client=fake_client)
        with pytest.raises(NotImplementedError, match="Use OpenHandsImporter"):
            runner.run("Fix the bug.")
