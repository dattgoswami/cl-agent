"""
Map OpenHands V1 conversation events into normalized CL-layer structures.

V1 event shape
--------------
Every serialized event is a JSON object with at least::

    id        – event UUID
    timestamp – ISO-8601 string (UTC)
    source    – "agent" | "user" | "environment" | "hook"

Event type is inferred by a combination of structural duck-typing (presence
of specific keys) and the top-level ``kind`` discriminator that some event
types carry:

    Structural (duck-typed on key presence):
      "action" key present (nested dict)  → ActionEvent
      "observation" key present            → ObservationEvent
      "llm_message" key present            → MessageEvent
      "system_prompt" key present          → SystemPromptEvent
      source=="agent" + "error" + "tool_call_id" → AgentErrorEvent
      "forgotten_event_ids" key present    → CondensationEvent (skipped)

    Top-level kind discriminator:
      kind == "ConversationStateUpdateEvent" → state signal (key/value)
      kind == "ConversationErrorEvent"        → fatal error
      kind == "ServerErrorEvent"             → fatal error
      kind == "HookExecutionEvent"           → hook event

    All remaining events (PauseEvent, CondensationRequestEvent, etc.)
    → silently skipped.

Action discriminators — nested ``action.kind`` (real V1 format, per
ActionBase<T> in frontend/src/types/v1/core/base/base.ts):
    ExecuteBashAction / TerminalAction → command_execution
    FileEditorAction / StrReplaceEditorAction / PlanningFileEditorAction
      (mutating commands only)           → file_change
    MCPToolAction                       → mcp_tool_call
    FinishAction                        → outcome.final_response + agent_message
    Browser* / ThinkAction / TaskTrackerAction / Glob* / Grep* → agent_message

Observation discriminator — nested ``observation.kind``.

The adapter reads ``action.get("kind")`` and falls back to
``action.get("type")`` so it works with both real V1 data and any
legacy/test data that uses ``type``.

Observation events are paired with their action via action_id and processed
in a two-pass approach: the observation index is built first, then actions
are processed and paired observations are looked up inline.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from cl_layer.episode.schema import EpisodeEvent, EpisodeOutcome

from .conversation_loader import OpenHandsConversation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TEXT_CHARS = 4_000
MAX_EXCERPT_CHARS = 2_000
MAX_THOUGHT_CHARS = 1_200

_MUTATING_FILE_COMMANDS = {"create", "str_replace", "insert", "undo_edit"}

_BROWSER_ACTION_TYPES = frozenset(
    {
        "BrowserNavigateAction",
        "BrowserClickAction",
        "BrowserTypeAction",
        "BrowserGetStateAction",
        "BrowserGetContentAction",
        "BrowserScrollAction",
        "BrowserGoBackAction",
        "BrowserListTabsAction",
        "BrowserSwitchTabAction",
        "BrowserCloseTabAction",
    }
)

_COMMAND_ACTION_TYPES = frozenset({"ExecuteBashAction", "TerminalAction"})

_FILE_ACTION_TYPES = frozenset(
    {"FileEditorAction", "StrReplaceEditorAction", "PlanningFileEditorAction"}
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OpenHandsMappingResult:
    events: list[EpisodeEvent]
    outcome: EpisodeOutcome
    patch_text: str | None
    patch_hash: str | None
    base_model_id: str | None
    tool_trace: list[dict] | None
    test_trace: list[dict] | None
    stdout_excerpt: str | None
    stderr_excerpt: str | None
    cost_tokens_prompt: int | None
    cost_tokens_completion: int | None


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

_FALLBACK_TS = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return _FALLBACK_TS
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return _FALLBACK_TS


# ---------------------------------------------------------------------------
# Text / content helpers
# ---------------------------------------------------------------------------

def _truncate(value: object, limit: int = MAX_TEXT_CHARS) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            value = str(value)
    assert isinstance(value, str)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...[truncated {len(value) - limit} chars]"


def _content_text(content: object, limit: int = MAX_TEXT_CHARS) -> str | None:
    """Extract plain text from a V1 content field.

    Handles:
    - plain ``str``
    - ``{"type": "text", "text": "..."}`` single TextContent dict
    - ``[{"type": "text", "text": "..."}]`` arrays
    """
    if isinstance(content, str):
        return _truncate(content, limit)
    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text")
        return _truncate(text, limit) if isinstance(text, str) else None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        combined = "\n".join(parts)
        return _truncate(combined, limit) if combined else None
    return None


def _thought_text(thought: object) -> str | None:
    """Extract agent thought from ActionEvent.thought (list of TextContent)."""
    return _content_text(thought, MAX_THOUGHT_CHARS)


def _message_text(llm_message: object) -> str | None:
    """Extract text from an LLM message dict."""
    if not isinstance(llm_message, dict):
        return None
    return _content_text(llm_message.get("content"))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _clean_path(path: object) -> str | None:
    """Normalize a file path: strip leading slash, remove whitespace."""
    if not isinstance(path, str) or not path.strip():
        return None
    return path.strip().lstrip("/")


# ---------------------------------------------------------------------------
# Patch / diff helpers
# ---------------------------------------------------------------------------

def _build_file_diff(
    path: str | None,
    old_content: str | None,
    new_content: str | None,
) -> str | None:
    """Construct a unified diff from old/new content strings.

    Returns ``None`` when new_content is absent (no evidence of change).
    Returns a creation diff when old_content is None/empty and new_content
    is present.
    """
    if new_content is None:
        return None
    clean = (path or "file").lstrip("/")
    old_lines = (old_content or "").splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    # Ensure trailing newlines so diff output is well-formed
    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{clean}",
            tofile=f"b/{clean}",
        )
    )
    return "".join(diff_lines) if diff_lines else None


def _hash_patch(patch_text: str | None) -> str | None:
    if not patch_text:
        return None
    digest = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Observation index
# ---------------------------------------------------------------------------

def _build_obs_index(raw_events: list[dict]) -> dict[str, dict]:
    """Build {action_id → raw_observation_event} from V1 event list."""
    index: dict[str, dict] = {}
    for ev in raw_events:
        if "observation" not in ev:
            continue
        action_id = ev.get("action_id")
        if action_id and str(action_id) not in index:
            index[str(action_id)] = ev
    return index


# ---------------------------------------------------------------------------
# Per-action-type handlers
# ---------------------------------------------------------------------------

def _handle_command_action(
    *,
    action: dict,
    obs_ev: dict | None,
    ev_id: str,
    ts: datetime,
    thought: str | None,
    events: list[EpisodeEvent],
    tool_trace: list[dict],
    test_trace: list[dict],
    test_outputs: list[str],
    failure_outputs: list[str],
) -> bool:
    """Emit a command_execution event.  Returns True if the command failed."""
    action_type = action.get("kind") or action.get("type") or ""
    command = action.get("command") or ""

    obs: dict = {}
    exit_code: int | None = None
    output: str | None = None
    timed_out = False

    if obs_ev and isinstance(obs_ev.get("observation"), dict):
        obs = obs_ev["observation"]
        exit_code_raw = obs.get("exit_code")
        if isinstance(exit_code_raw, int):
            exit_code = exit_code_raw
        output = _content_text(obs.get("content"), MAX_TEXT_CHARS)
        timed_out = bool(obs.get("timeout"))

    # Infer whether this looked like a test command
    is_test = _is_test_command(command)

    metadata = obs.get("metadata")
    working_dir: str | None = None
    if isinstance(metadata, dict):
        working_dir = metadata.get("working_dir") or None

    payload: dict = {
        "source": "openhands_action",
        "openhands_action_type": action_type,
        "event_id": ev_id,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "is_input": action.get("is_input"),
        "working_dir": working_dir,
        "output": _truncate(output, MAX_TEXT_CHARS),
        "thought": thought,
        "is_test_command": is_test,
    }

    events.append(EpisodeEvent(kind="command_execution", timestamp=ts, payload=payload))
    tool_trace.append(payload)

    if is_test and output:
        test_record: dict = {
            "command": command,
            "exit_code": exit_code,
            "passed": exit_code == 0 if exit_code is not None else None,
            "source": "openhands_action",
            "event_id": ev_id,
        }
        test_trace.append(test_record)
        test_outputs.append(output)

    if exit_code is not None and exit_code != 0:
        if output:
            failure_outputs.append(output)
        return True
    if timed_out:
        if output:
            failure_outputs.append(output)
        return True
    return False


def _handle_file_action(
    *,
    action: dict,
    obs_ev: dict | None,
    ev_id: str,
    ts: datetime,
    thought: str | None,
    events: list[EpisodeEvent],
    file_diffs: list[str],
) -> str | None:
    """Emit a file_change event.  Returns the cleaned file path or None."""
    action_type = action.get("kind") or action.get("type") or ""
    command = (action.get("command") or "").strip().lower()

    if command not in _MUTATING_FILE_COMMANDS:
        # "view" → no change; skip
        return None

    path = _clean_path(action.get("path"))

    # Extract diff from paired observation
    old_content: str | None = None
    new_content: str | None = None
    obs_error: str | None = None

    if obs_ev and isinstance(obs_ev.get("observation"), dict):
        obs: dict = obs_ev["observation"]
        old_content = obs.get("old_content") if isinstance(obs.get("old_content"), str) else None
        new_content = obs.get("new_content") if isinstance(obs.get("new_content"), str) else None
        obs_error = obs.get("error") if isinstance(obs.get("error"), str) else None

    diff = _build_file_diff(path, old_content, new_content)
    if diff:
        file_diffs.append(diff)

    payload: dict = {
        "source": "openhands_action",
        "openhands_action_type": action_type,
        "event_id": ev_id,
        "operation": command,
        "path": path,
        "patch_hash": _hash_patch(diff) if diff else None,
        "patch_chars": len(diff) if diff else 0,
        "thought": thought,
        "obs_error": obs_error,
    }

    events.append(EpisodeEvent(kind="file_change", timestamp=ts, payload=payload))
    return path


def _handle_mcp_action(
    *,
    action: dict,
    obs_ev: dict | None,
    ev_id: str,
    ts: datetime,
    thought: str | None,
    events: list[EpisodeEvent],
) -> None:
    """Emit a mcp_tool_call event."""
    data: dict = action.get("data") if isinstance(action.get("data"), dict) else {}
    tool_name: str | None = data.get("tool_name") or data.get("name")

    is_error = False
    obs_tool_name: str | None = None
    if obs_ev and isinstance(obs_ev.get("observation"), dict):
        obs = obs_ev["observation"]
        is_error = bool(obs.get("is_error"))
        obs_tool_name = obs.get("tool_name") or None

    payload: dict = {
        "source": "openhands_action",
        "openhands_action_type": "MCPToolAction",
        "event_id": ev_id,
        "tool_name": tool_name or obs_tool_name,
        "data": data,
        "is_error": is_error,
        "thought": thought,
    }
    events.append(EpisodeEvent(kind="mcp_tool_call", timestamp=ts, payload=payload))


def _handle_browser_action(
    *,
    action: dict,
    obs_ev: dict | None,
    ev_id: str,
    ts: datetime,
    thought: str | None,
    events: list[EpisodeEvent],
) -> None:
    """Emit an agent_message for a browser/platform action."""
    action_type = action.get("kind") or action.get("type") or ""
    obs_output: str | None = None
    if obs_ev and isinstance(obs_ev.get("observation"), dict):
        obs = obs_ev["observation"]
        obs_output = _truncate(obs.get("output"), MAX_EXCERPT_CHARS)

    payload: dict = {
        "source": "openhands_browser_action",
        "openhands_action_type": action_type,
        "event_id": ev_id,
        "url": action.get("url"),
        "text": action.get("text"),
        "index": action.get("index"),
        "new_tab": action.get("new_tab"),
        "thought": thought,
        "obs_output": obs_output,
    }
    events.append(EpisodeEvent(kind="agent_message", timestamp=ts, payload=payload))


# ---------------------------------------------------------------------------
# Test-command detection
# ---------------------------------------------------------------------------

_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|unittest|npm\s+(run\s+)?test|pnpm\s+test|yarn\s+test|"
    r"vitest|go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|"
    r"ruff|flake8|mypy|eslint|tsc)\b",
    re.IGNORECASE,
)


def _is_test_command(command: str | None) -> bool:
    return bool(command and _TEST_COMMAND_RE.search(command))


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------

def _base_model_from_meta(meta: dict) -> str | None:
    # llm_model is the canonical field in AppConversationInfo / meta.json
    for key in ("llm_model", "model", "base_model", "base_model_id"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    llm_config = meta.get("llm_config")
    if isinstance(llm_config, dict):
        for key in ("model", "model_name"):
            value = llm_config.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _task_from_meta(meta: dict) -> str | None:
    for key in ("title", "task", "initial_message", "task_description", "problem_statement"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# ---------------------------------------------------------------------------
# Outcome status derivation
# ---------------------------------------------------------------------------

def _derive_outcome_status(
    *,
    events: list[EpisodeEvent],
    final_response: str | None,
    had_command_failure: bool,
    had_fatal_error: bool,
    conversation_finished: bool,
    conversation_in_progress: bool,
) -> tuple[str, str | None]:
    """Derive (EpisodeStatus, escalation_reason | None) from observable signals."""
    # FinishAction is the canonical success signal
    if final_response:
        if had_command_failure:
            return "partial", None
        return "completed", None

    # Explicit conversation-finished state can be the only emitted event.
    if conversation_finished:
        if had_command_failure:
            return "partial", None
        return "completed", None

    if had_fatal_error:
        if events:
            return "partial", "agent error encountered"
        return "failed", "agent error without any completed events"

    if had_command_failure:
        return "partial", None

    if conversation_in_progress:
        return "partial", "conversation did not reach a terminal state"

    if not events:
        return "failed", "no parseable OpenHands events"

    # Events present, no explicit finish/error state. Do not infer success.
    return "partial", "no OpenHands finish signal"


# ---------------------------------------------------------------------------
# Token count extraction
# ---------------------------------------------------------------------------

def _extract_token_counts(stats_value: dict) -> tuple[int | None, int | None]:
    """Extract (prompt_tokens, completion_tokens) from a ConversationStats value dict."""
    try:
        usage = (
            stats_value
            .get("usage_to_metrics", {})
            .get("agent", {})
            .get("accumulated_token_usage", {})
        )
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        return (
            int(prompt) if isinstance(prompt, (int, float)) else None,
            int(completion) if isinstance(completion, (int, float)) else None,
        )
    except (AttributeError, TypeError, ValueError):
        return None, None


def _extract_metrics_snapshot_token_counts(metrics_value: dict) -> tuple[int | None, int | None]:
    """Extract token counts from AppConversationInfo.metrics in meta.json."""
    try:
        usage = metrics_value.get("accumulated_token_usage", {})
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        return (
            int(prompt) if isinstance(prompt, (int, float)) else None,
            int(completion) if isinstance(completion, (int, float)) else None,
        )
    except (AttributeError, TypeError, ValueError):
        return None, None


# ---------------------------------------------------------------------------
# Main mapping entry point
# ---------------------------------------------------------------------------

def map_conversation(conv: OpenHandsConversation) -> OpenHandsMappingResult:
    """Map a loaded OpenHands V1 conversation into a normalized mapping result.

    This is the primary public function.  It performs two passes over the
    event list:

    1. Build an ``{action_id → observation_event}`` index.
    2. Process events in order, pairing actions with their observations.
    """
    obs_index = _build_obs_index(conv.events)

    # Accumulation state
    cl_events: list[EpisodeEvent] = []
    files_touched: list[str] = []
    file_diffs: list[str] = []
    tool_trace: list[dict] = []
    test_trace: list[dict] = []
    test_outputs: list[str] = []
    failure_outputs: list[str] = []

    final_response: str | None = None
    had_command_failure = False
    had_fatal_error = False
    conversation_finished = False
    conversation_in_progress = False
    cost_tokens_prompt: int | None = None
    cost_tokens_completion: int | None = None

    for raw_ev in conv.events:
        ev_id = str(raw_ev.get("id") or "")
        ts = _parse_timestamp(raw_ev.get("timestamp"))
        raw_kind = raw_ev.get("kind") or ""

        # ------------------------------------------------------------------ #
        # 1. Action events
        # ------------------------------------------------------------------ #
        action = raw_ev.get("action")
        if isinstance(action, dict):
            action_type: str = action.get("kind") or action.get("type") or ""
            thought = _thought_text(raw_ev.get("thought"))
            obs_ev = obs_index.get(ev_id) if ev_id else None

            if action_type in _COMMAND_ACTION_TYPES:
                failed = _handle_command_action(
                    action=action,
                    obs_ev=obs_ev,
                    ev_id=ev_id,
                    ts=ts,
                    thought=thought,
                    events=cl_events,
                    tool_trace=tool_trace,
                    test_trace=test_trace,
                    test_outputs=test_outputs,
                    failure_outputs=failure_outputs,
                )
                if failed:
                    had_command_failure = True

            elif action_type in _FILE_ACTION_TYPES:
                path = _handle_file_action(
                    action=action,
                    obs_ev=obs_ev,
                    ev_id=ev_id,
                    ts=ts,
                    thought=thought,
                    events=cl_events,
                    file_diffs=file_diffs,
                )
                if path:
                    files_touched.append(path)

            elif action_type == "MCPToolAction":
                _handle_mcp_action(
                    action=action,
                    obs_ev=obs_ev,
                    ev_id=ev_id,
                    ts=ts,
                    thought=thought,
                    events=cl_events,
                )

            elif action_type == "FinishAction":
                message = action.get("message")
                if isinstance(message, str) and message.strip():
                    final_response = _truncate(message)
                # Also emit as agent_message so the thought is preserved
                if thought or message:
                    cl_events.append(
                        EpisodeEvent(
                            kind="agent_message",
                            timestamp=ts,
                            payload={
                                "source": "openhands_finish",
                                "event_id": ev_id,
                                "message": _truncate(message) if isinstance(message, str) else None,
                                "thought": thought,
                            },
                        )
                    )

            elif action_type in _BROWSER_ACTION_TYPES:
                _handle_browser_action(
                    action=action,
                    obs_ev=obs_ev,
                    ev_id=ev_id,
                    ts=ts,
                    thought=thought,
                    events=cl_events,
                )

            elif action_type == "ThinkAction":
                think_text = action.get("thought")
                if think_text or thought:
                    cl_events.append(
                        EpisodeEvent(
                            kind="agent_message",
                            timestamp=ts,
                            payload={
                                "source": "openhands_think",
                                "event_id": ev_id,
                                "thought": _truncate(
                                    think_text if isinstance(think_text, str) else thought,
                                    MAX_THOUGHT_CHARS,
                                ),
                            },
                        )
                    )

            elif action_type in {"TaskTrackerAction", "GlobAction", "GrepAction"}:
                pattern = action.get("pattern") or action.get("command")
                cl_events.append(
                    EpisodeEvent(
                        kind="agent_message",
                        timestamp=ts,
                        payload={
                            "source": "openhands_action",
                            "openhands_action_type": action_type,
                            "event_id": ev_id,
                            "pattern": pattern,
                            "thought": thought,
                        },
                    )
                )

            elif action_type:
                # Unknown action type — preserve as agent_message
                cl_events.append(
                    EpisodeEvent(
                        kind="agent_message",
                        timestamp=ts,
                        payload={
                            "source": "openhands_action",
                            "openhands_action_type": action_type,
                            "event_id": ev_id,
                            "thought": thought,
                        },
                    )
                )
            continue  # action event fully handled

        # ------------------------------------------------------------------ #
        # 2. Observation events — paired with actions above; skip here to
        #    avoid double-counting.  Unpaired observations (no action_id or
        #    action was not captured) are ignored.
        # ------------------------------------------------------------------ #
        if "observation" in raw_ev:
            continue

        # ------------------------------------------------------------------ #
        # 3. MessageEvent
        # ------------------------------------------------------------------ #
        if "llm_message" in raw_ev:
            llm_message = raw_ev.get("llm_message")
            role = "unknown"
            if isinstance(llm_message, dict):
                role = llm_message.get("role") or "unknown"
            text = _message_text(llm_message)
            cl_events.append(
                EpisodeEvent(
                    kind="agent_message",
                    timestamp=ts,
                    payload={
                        "source": "openhands_message",
                        "event_id": ev_id,
                        "role": role,
                        "text": _truncate(text),
                        "activated_microagents": raw_ev.get("activated_microagents") or [],
                    },
                )
            )
            continue

        # ------------------------------------------------------------------ #
        # 4. SystemEvent
        # ------------------------------------------------------------------ #
        if "system_prompt" in raw_ev:
            system_prompt = raw_ev.get("system_prompt")
            cl_events.append(
                EpisodeEvent(
                    kind="agent_message",
                    timestamp=ts,
                    payload={
                        "source": "openhands_system",
                        "event_id": ev_id,
                        "system_prompt": _content_text(system_prompt),
                    },
                )
            )
            continue

        # ------------------------------------------------------------------ #
        # 5. ConversationStateUpdateEvent — update state flags / token counts
        # ------------------------------------------------------------------ #
        if raw_kind == "ConversationStateUpdateEvent":
            key = raw_ev.get("key")
            value = raw_ev.get("value")
            if key == "execution_status" and isinstance(value, str):
                normalized = value.strip().lower()
                if normalized == "finished":
                    conversation_finished = True
                elif normalized in {"error", "stuck"}:
                    had_fatal_error = True
                else:
                    conversation_in_progress = True
            elif key == "stats" and isinstance(value, dict):
                # value is ConversationStats: {usage_to_metrics: {agent: {accumulated_token_usage: ...}}}
                prompt_tok, completion_tok = _extract_token_counts(value)
                if prompt_tok is not None:
                    cost_tokens_prompt = prompt_tok
                if completion_tok is not None:
                    cost_tokens_completion = completion_tok
            elif key == "full_state" and isinstance(value, dict):
                # value is ConversationState: {execution_status, stats?: ConversationStats}
                exec_status = value.get("execution_status")
                if isinstance(exec_status, str):
                    normalized = exec_status.strip().lower()
                    if normalized == "finished":
                        conversation_finished = True
                    elif normalized in {"error", "stuck"}:
                        had_fatal_error = True
                    else:
                        conversation_in_progress = True
                nested_stats = value.get("stats")
                if isinstance(nested_stats, dict):
                    prompt_tok, completion_tok = _extract_token_counts(nested_stats)
                    if prompt_tok is not None:
                        cost_tokens_prompt = prompt_tok
                    if completion_tok is not None:
                        cost_tokens_completion = completion_tok
            continue

        # ------------------------------------------------------------------ #
        # 5a. ConversationErrorEvent / ServerErrorEvent — fatal, emit event
        # ------------------------------------------------------------------ #
        if raw_kind in {"ConversationErrorEvent", "ServerErrorEvent"}:
            had_fatal_error = True
            cl_events.append(
                EpisodeEvent(
                    kind="agent_message",
                    timestamp=ts,
                    payload={
                        "source": "openhands_error",
                        "event_id": ev_id,
                        "error_kind": raw_kind,
                        "error": _truncate(
                            str(raw_ev.get("detail") or raw_ev.get("message") or ""),
                            MAX_EXCERPT_CHARS,
                        ),
                    },
                )
            )
            continue

        # ------------------------------------------------------------------ #
        # 6. HookExecutionEvent
        # ------------------------------------------------------------------ #
        if raw_kind == "HookExecutionEvent":
            cl_events.append(
                EpisodeEvent(
                    kind="agent_message",
                    timestamp=ts,
                    payload={
                        "source": "openhands_hook",
                        "event_id": ev_id,
                        "hook_event_type": raw_ev.get("hook_event_type"),
                        "hook_command": raw_ev.get("hook_command"),
                        "success": raw_ev.get("success"),
                        "blocked": raw_ev.get("blocked"),
                        "exit_code": raw_ev.get("exit_code"),
                        "reason": raw_ev.get("reason"),
                        "tool_name": raw_ev.get("tool_name"),
                        "action_id": raw_ev.get("action_id"),
                        "message_id": raw_ev.get("message_id"),
                        "stdout": _truncate(raw_ev.get("stdout"), MAX_EXCERPT_CHARS),
                        "stderr": _truncate(raw_ev.get("stderr"), MAX_EXCERPT_CHARS),
                        "error": _truncate(raw_ev.get("error"), MAX_EXCERPT_CHARS),
                        "additional_context": _truncate(raw_ev.get("additional_context"), MAX_EXCERPT_CHARS),
                        "hook_input": raw_ev.get("hook_input"),
                    },
                )
            )
            continue

        # ------------------------------------------------------------------ #
        # 7. AgentErrorEvent (source=agent + error key + tool_call_id)
        # ------------------------------------------------------------------ #
        if raw_ev.get("source") == "agent" and "error" in raw_ev and "tool_call_id" in raw_ev:
            had_fatal_error = True
            cl_events.append(
                EpisodeEvent(
                    kind="agent_message",
                    timestamp=ts,
                    payload={
                        "source": "openhands_agent_error",
                        "event_id": ev_id,
                        "error": _truncate(str(raw_ev.get("error") or ""), MAX_EXCERPT_CHARS),
                    },
                )
            )
            continue

        # CondensationEvent, PauseEvent, UserRejectObservation → silently skip

    # ------------------------------------------------------------------ #
    # Build combined patch
    # ------------------------------------------------------------------ #
    patch_text: str | None = None
    if file_diffs:
        combined = "\n".join(file_diffs)
        if combined.strip():
            patch_text = combined

    patch_hash = _hash_patch(patch_text)

    # ------------------------------------------------------------------ #
    # Derive outcome
    # ------------------------------------------------------------------ #
    outcome_status, escalation_reason = _derive_outcome_status(
        events=cl_events,
        final_response=final_response,
        had_command_failure=had_command_failure,
        had_fatal_error=had_fatal_error,
        conversation_finished=conversation_finished,
        conversation_in_progress=conversation_in_progress,
    )

    outcome = EpisodeOutcome(
        status=outcome_status,  # type: ignore[arg-type]
        tests_passed=None,  # filled by evaluator
        verification_summary=None,
        escalation_reason=escalation_reason,
        files_touched=sorted(dict.fromkeys(files_touched)),
        final_response=final_response,
    )

    # ------------------------------------------------------------------ #
    # base_model_id from meta
    # ------------------------------------------------------------------ #
    base_model_id = _base_model_from_meta(conv.meta)

    meta_metrics = conv.meta.get("metrics")
    if isinstance(meta_metrics, dict):
        meta_prompt, meta_completion = _extract_metrics_snapshot_token_counts(meta_metrics)
        if cost_tokens_prompt is None and meta_prompt is not None:
            cost_tokens_prompt = meta_prompt
        if cost_tokens_completion is None and meta_completion is not None:
            cost_tokens_completion = meta_completion

    return OpenHandsMappingResult(
        events=cl_events,
        outcome=outcome,
        patch_text=patch_text,
        patch_hash=patch_hash,
        base_model_id=base_model_id,
        tool_trace=tool_trace or None,
        test_trace=test_trace or None,
        stdout_excerpt=_truncate(test_outputs[-1], MAX_EXCERPT_CHARS) if test_outputs else None,
        stderr_excerpt=_truncate(failure_outputs[-1], MAX_EXCERPT_CHARS) if failure_outputs else None,
        cost_tokens_prompt=cost_tokens_prompt,
        cost_tokens_completion=cost_tokens_completion,
    )
