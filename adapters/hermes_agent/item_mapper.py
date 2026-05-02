from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cl_layer.episode.schema import EpisodeEvent, EpisodeOutcome

MAX_TEXT_CHARS = 4000
MAX_ARG_TEXT_CHARS = 700

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_XML_BLOCK_RE = re.compile(r"<(?:think|tool_call|tool_response)>\s*.*?\s*</(?:think|tool_call|tool_response)>", re.DOTALL)
_PATCH_FILE_RE = re.compile(r"^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$", re.MULTILINE)
_FAILURE_TEXT_RE = re.compile(
    r"\b(failed|failure|error|aborted|blocked|could not|couldn't|cannot complete)\b",
    re.IGNORECASE,
)
_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|unittest|npm\s+(run\s+)?test|pnpm\s+test|yarn\s+test|"
    r"vitest|go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|npm\s+run\s+check)\b"
)

COMMAND_TOOLS = {"terminal", "bash", "shell", "execute_code", "code_execution"}
FILE_TOOLS = {"read_file", "write_file", "patch", "search_files", "read", "write", "edit"}
MEMORY_TOOLS = {"memory"}
SKILL_TOOLS = {"skills_list", "skill_view", "skill", "skills"}
SESSION_SEARCH_TOOLS = {"session_search"}
SUBAGENT_TOOLS = {"delegate", "spawn_agent", "subagent", "mixture_of_agents"}


@dataclass
class HermesMappingResult:
    events: list[EpisodeEvent]
    outcome: EpisodeOutcome
    base_model_id: str | None
    patch_text: str | None
    patch_hash: str | None
    tool_trace: list[dict] | None
    test_trace: list[dict] | None
    stdout_excerpt: str | None
    stderr_excerpt: str | None
    cost_tokens_prompt: int | None
    cost_tokens_completion: int | None


def _truncate(text: object, limit: int = MAX_TEXT_CHARS) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _decode_jsonish(text: object) -> Any:
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    try:
        value, _idx = json.JSONDecoder().raw_decode(stripped)
        return value
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def _extract_blocks(pattern: re.Pattern[str], text: str) -> list[Any]:
    blocks: list[Any] = []
    for match in pattern.finditer(text or ""):
        decoded = _decode_jsonish(match.group(1))
        if decoded is not None:
            blocks.append(decoded)
    return blocks


def _message_text(value: str) -> tuple[str | None, int]:
    thinking_chars = sum(len(match.group(1)) for match in _THINK_RE.finditer(value or ""))
    without_blocks = _XML_BLOCK_RE.sub("", value or "")
    text = without_blocks.strip()
    return _truncate(text or None), thinking_chars


def _compact_arguments(tool_name: str | None, arguments: object) -> dict:
    if not isinstance(arguments, dict):
        return {}

    if tool_name in {"terminal", "bash", "shell"}:
        return {
            "command": _truncate(arguments.get("command"), MAX_ARG_TEXT_CHARS),
            "background": arguments.get("background"),
            "timeout": arguments.get("timeout"),
            "workdir": arguments.get("workdir"),
        }
    if tool_name in {"execute_code", "code_execution"}:
        code = arguments.get("code")
        return {
            "code_chars": len(code) if isinstance(code, str) else None,
            "code_excerpt": _truncate(code, MAX_ARG_TEXT_CHARS),
        }
    if tool_name in {"read_file", "read"}:
        return {
            "path": _path_from_args(arguments),
            "offset": arguments.get("offset"),
            "limit": arguments.get("limit"),
        }
    if tool_name in {"write_file", "write"}:
        content = arguments.get("content")
        return {
            "path": _path_from_args(arguments),
            "content_chars": len(content) if isinstance(content, str) else None,
        }
    if tool_name in {"patch", "edit"}:
        patch_text = arguments.get("patch")
        return {
            "mode": arguments.get("mode"),
            "path": _path_from_args(arguments),
            "replace_all": arguments.get("replace_all"),
            "old_string_chars": len(arguments.get("old_string")) if isinstance(arguments.get("old_string"), str) else None,
            "new_string_chars": len(arguments.get("new_string")) if isinstance(arguments.get("new_string"), str) else None,
            "patch_chars": len(patch_text) if isinstance(patch_text, str) else None,
            "patch_excerpt": _truncate(patch_text, MAX_ARG_TEXT_CHARS),
        }
    if tool_name == "memory":
        return {
            "action": arguments.get("action"),
            "target": arguments.get("target"),
            "content_chars": len(arguments.get("content")) if isinstance(arguments.get("content"), str) else None,
            "old_text_chars": len(arguments.get("old_text")) if isinstance(arguments.get("old_text"), str) else None,
        }
    if tool_name in SKILL_TOOLS:
        return {
            "name": arguments.get("name"),
            "category": arguments.get("category"),
            "file_path": arguments.get("file_path"),
        }

    compact: dict[str, object] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            compact[key] = _truncate(value, MAX_ARG_TEXT_CHARS)
        elif isinstance(value, (int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = {"type": "list", "length": len(value)}
        elif isinstance(value, dict):
            compact[key] = {"type": "object", "keys": sorted(str(k) for k in value.keys())[:20]}
        else:
            compact[key] = str(type(value).__name__)
    return compact


def _path_from_args(arguments: object) -> str | None:
    if not isinstance(arguments, dict):
        return None
    path = arguments.get("path") or arguments.get("file_path")
    return str(path) if path else None


def _paths_from_patch_text(patch_text: str | None) -> list[str]:
    if not patch_text:
        return []
    return [match.group(1).strip() for match in _PATCH_FILE_RE.finditer(patch_text)]


def _result_content(response: dict) -> Any:
    content = response.get("content")
    return _decode_jsonish(content) if isinstance(content, str) else content


def _result_error(result: Any) -> str | None:
    if isinstance(result, dict):
        error = result.get("error")
        if error:
            return str(error)
        if result.get("success") is False:
            return "success=false"
        if result.get("status") in {"error", "blocked", "approval_required", "timeout"}:
            return str(result.get("status"))
    return None


def _result_output(result: Any) -> str | None:
    if isinstance(result, dict):
        output = result.get("output")
        if isinstance(output, str):
            return _truncate(output)
        content = result.get("content")
        if isinstance(content, str):
            return _truncate(content)
    if isinstance(result, str):
        return _truncate(result)
    return None


def _is_success_result(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("success") is False:
            return False
        if result.get("error"):
            return False
        if result.get("status") in {"error", "blocked", "approval_required", "timeout"}:
            return False
    return True


def _is_test_command(command: str | None) -> bool:
    return bool(command and _TEST_COMMAND_RE.search(command))


def _parse_mcp_name(tool_name: str) -> tuple[str | None, str | None]:
    if not tool_name.startswith("mcp_"):
        return None, None
    remainder = tool_name[len("mcp_") :]
    utility_suffixes = ("_list_resources", "_read_resource", "_list_prompts", "_get_prompt")
    for suffix in utility_suffixes:
        if remainder.endswith(suffix):
            return remainder[: -len(suffix)] or None, suffix[1:]
    # Hermes flattens MCP names to mcp_<server>_<tool>. Without the original
    # server registry this is ambiguous when server names contain underscores,
    # so preserve hermes_tool_name in the payload and keep this best-effort.
    parts = remainder.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return remainder or None, None


def _append_agent_message(events: list[EpisodeEvent], timestamp: datetime, payload: dict) -> None:
    events.append(EpisodeEvent(kind="agent_message", timestamp=timestamp, payload=payload))


def _match_pending_call(pending: list[dict], response: dict) -> dict | None:
    name = response.get("name")
    if isinstance(name, str):
        for idx, call in enumerate(pending):
            if call.get("name") == name:
                return pending.pop(idx)
    if pending:
        return pending.pop(0)
    return None


def _map_command_tool(
    *,
    events: list[EpisodeEvent],
    timestamp: datetime,
    call: dict,
    response: dict,
    result: Any,
    test_trace: list[dict],
    command_outputs: list[str],
) -> bool:
    tool_name = call.get("name") or response.get("name")
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    is_code = tool_name in {"execute_code", "code_execution"}
    command = None
    code_excerpt = None
    if is_code:
        code = args.get("code") if isinstance(args, dict) else None
        code_excerpt = _truncate(code, 1200)
        command = "python <execute_code>"
    elif isinstance(args, dict):
        command = args.get("command")

    exit_code: int | None = None
    if isinstance(result, dict):
        raw_exit = result.get("exit_code")
        if isinstance(raw_exit, int):
            exit_code = raw_exit
        elif is_code:
            exit_code = 0 if result.get("status") == "success" and not result.get("error") else 1

    error = _result_error(result)
    failed = bool(error) or (exit_code is not None and exit_code != 0)
    output = _result_output(result)
    if output:
        command_outputs.append(output)
    if _is_test_command(command) and exit_code is not None:
        test_trace.append({"command": command, "exit_code": exit_code, "source": "hermes_trajectory"})

    payload = {
        "source": "hermes_trajectory",
        "tool_name": tool_name,
        "tool_call_id": response.get("tool_call_id"),
        "command": command,
        "code_excerpt": code_excerpt,
        "exit_code": exit_code,
        "status": result.get("status") if isinstance(result, dict) else None,
        "duration_seconds": result.get("duration_seconds") if isinstance(result, dict) else None,
        "workdir": args.get("workdir") if isinstance(args, dict) else None,
        "timeout": args.get("timeout") if isinstance(args, dict) else None,
        "output": output,
        "error": error,
        "result": result if isinstance(result, dict) else _truncate(result),
    }
    events.append(EpisodeEvent(kind="command_execution", timestamp=timestamp, payload=payload))
    return failed


def _map_file_tool(
    *,
    events: list[EpisodeEvent],
    timestamp: datetime,
    call: dict,
    response: dict,
    result: Any,
    files_touched: set[str],
    diffs: list[str],
) -> None:
    tool_name = call.get("name") or response.get("name")
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    result_dict = result if isinstance(result, dict) else {}
    operation_map = {
        "read": "read",
        "read_file": "read",
        "write": "write",
        "write_file": "write",
        "edit": "edit",
        "patch": "patch",
        "search_files": "search",
    }
    operation = operation_map.get(str(tool_name), str(tool_name))
    mutating = operation in {"write", "edit", "patch"}
    error = _result_error(result)
    success = error is None and _is_success_result(result)

    paths: list[str] = []
    direct_path = _path_from_args(args)
    if direct_path:
        paths.append(direct_path)
    for key in ("files_modified", "files_created", "files_deleted"):
        values = result_dict.get(key)
        if isinstance(values, list):
            paths.extend(str(v) for v in values if v)
    paths.extend(_paths_from_patch_text(args.get("patch") if isinstance(args, dict) else None))
    paths = sorted(dict.fromkeys(paths))

    if mutating and success:
        files_touched.update(paths)

    diff = result_dict.get("diff") if isinstance(result_dict.get("diff"), str) else None
    patch_argument = args.get("patch") if isinstance(args, dict) and isinstance(args.get("patch"), str) else None
    if diff:
        diffs.append(diff)
    elif patch_argument and success:
        diffs.append(patch_argument)

    events.append(
        EpisodeEvent(
            kind="file_change",
            timestamp=timestamp,
            payload={
                "source": "hermes_trajectory",
                "tool_name": tool_name,
                "tool_call_id": response.get("tool_call_id"),
                "operation": operation,
                "path": paths[0] if paths else direct_path,
                "paths": paths,
                "mutating": mutating,
                "is_error": error is not None,
                "error": error,
                "result": result_dict if isinstance(result, dict) else _truncate(result),
                "diff": diff,
                "patch_excerpt": _truncate(patch_argument),
            },
        )
    )


def _map_mcp_tool(
    *,
    events: list[EpisodeEvent],
    timestamp: datetime,
    call: dict,
    response: dict,
    result: Any,
) -> None:
    tool_name = str(call.get("name") or response.get("name") or "")
    server, tool = _parse_mcp_name(tool_name)
    error = _result_error(result)
    events.append(
        EpisodeEvent(
            kind="mcp_tool_call",
            timestamp=timestamp,
            payload={
                "source": "hermes_trajectory",
                "hermes_tool_name": tool_name,
                "server": server,
                "tool": tool,
                "tool_call_id": response.get("tool_call_id"),
                "arguments": _compact_arguments(tool_name, call.get("arguments")),
                "is_error": error is not None,
                "error": error,
                "result": result if isinstance(result, dict) else _truncate(result),
            },
        )
    )


def _map_evidence_tool(
    *,
    events: list[EpisodeEvent],
    timestamp: datetime,
    call: dict,
    response: dict,
    result: Any,
    event_type: str,
) -> None:
    tool_name = call.get("name") or response.get("name")
    error = _result_error(result)
    _append_agent_message(
        events,
        timestamp,
        {
            "role": "tool",
            "hermes_event_type": event_type,
            "tool_name": tool_name,
            "tool_call_id": response.get("tool_call_id"),
            "arguments": _compact_arguments(str(tool_name), call.get("arguments")),
            "is_error": error is not None,
            "error": error,
            "result": result if isinstance(result, dict) else _truncate(result),
        },
    )


def map_hermes_conversations(
    conversations: list[dict],
    *,
    timestamp: datetime,
    metadata: dict | None = None,
    completed: bool | None = None,
    partial: bool | None = None,
    error: str | None = None,
) -> HermesMappingResult:
    events: list[EpisodeEvent] = []
    pending_calls: list[dict] = []
    tool_trace: list[dict] = []
    test_trace: list[dict] = []
    command_outputs: list[str] = []
    stderr_outputs: list[str] = []
    files_touched: set[str] = set()
    diffs: list[str] = []
    final_response: str | None = None
    had_command_failure = False
    had_tool_error = False
    # test_trace keeps every test-like command; tests_passed summarizes the
    # latest one to match CL's coarse outcome field.
    last_test_exit_code: int | None = None

    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            continue
        native_from = message.get("from") or message.get("role")
        value = message.get("value") if isinstance(message.get("value"), str) else message.get("content")
        value = value if isinstance(value, str) else ""

        if native_from in {"system", "human", "user", "gpt", "assistant"}:
            role_map = {"human": "user", "gpt": "assistant"}
            role = role_map.get(str(native_from), str(native_from))
            text, thinking_chars = _message_text(value)
            tool_calls = _extract_blocks(_TOOL_CALL_RE, value)
            compact_calls: list[dict] = []

            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_name = call.get("name")
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                call_record = {
                    "name": tool_name,
                    "arguments": arguments,
                    "message_index": index,
                }
                pending_calls.append(call_record)
                compact = {
                    "name": tool_name,
                    "arguments": _compact_arguments(str(tool_name), arguments),
                }
                compact_calls.append(compact)
                tool_trace.append({"phase": "call", **compact})

            if role == "assistant" and text:
                final_response = text

            _append_agent_message(
                events,
                timestamp,
                {
                    "role": role,
                    "text": text,
                    "thinking_chars": thinking_chars,
                    "tool_calls": compact_calls,
                    "message_index": index,
                    "source": "hermes_trajectory",
                },
            )
            continue

        if native_from == "tool":
            responses = _extract_blocks(_TOOL_RESPONSE_RE, value)
            if not responses:
                responses = [_decode_jsonish(value)]
            for response in responses:
                if not isinstance(response, dict):
                    _append_agent_message(
                        events,
                        timestamp,
                        {
                            "role": "tool",
                            "hermes_event_type": "tool_result",
                            "text": _truncate(response),
                            "message_index": index,
                            "source": "hermes_trajectory",
                        },
                    )
                    continue

                call = _match_pending_call(pending_calls, response) or {
                    "name": response.get("name"),
                    "arguments": {},
                    "message_index": None,
                }
                tool_name = str(call.get("name") or response.get("name") or "")
                result = _result_content(response)
                error_text = _result_error(result)
                if error_text:
                    had_tool_error = True
                    stderr_outputs.append(error_text)
                tool_trace.append(
                    {
                        "phase": "result",
                        "tool_name": tool_name,
                        "tool_call_id": response.get("tool_call_id"),
                        "is_error": error_text is not None,
                        "error": error_text,
                        "result": result if isinstance(result, dict) else _truncate(result, MAX_ARG_TEXT_CHARS),
                    }
                )

                if tool_name in COMMAND_TOOLS:
                    failed = _map_command_tool(
                        events=events,
                        timestamp=timestamp,
                        call=call,
                        response=response,
                        result=result,
                        test_trace=test_trace,
                        command_outputs=command_outputs,
                    )
                    had_command_failure = had_command_failure or failed
                    if test_trace and isinstance(test_trace[-1].get("exit_code"), int):
                        last_test_exit_code = test_trace[-1]["exit_code"]
                elif tool_name in FILE_TOOLS:
                    _map_file_tool(
                        events=events,
                        timestamp=timestamp,
                        call=call,
                        response=response,
                        result=result,
                        files_touched=files_touched,
                        diffs=diffs,
                    )
                elif tool_name.startswith("mcp_"):
                    _map_mcp_tool(events=events, timestamp=timestamp, call=call, response=response, result=result)
                elif tool_name in MEMORY_TOOLS:
                    _map_evidence_tool(
                        events=events,
                        timestamp=timestamp,
                        call=call,
                        response=response,
                        result=result,
                        event_type="memory_event",
                    )
                elif tool_name in SKILL_TOOLS:
                    _map_evidence_tool(
                        events=events,
                        timestamp=timestamp,
                        call=call,
                        response=response,
                        result=result,
                        event_type="skill_event",
                    )
                elif tool_name in SESSION_SEARCH_TOOLS:
                    _map_evidence_tool(
                        events=events,
                        timestamp=timestamp,
                        call=call,
                        response=response,
                        result=result,
                        event_type="session_search_event",
                    )
                elif tool_name in SUBAGENT_TOOLS:
                    _map_evidence_tool(
                        events=events,
                        timestamp=timestamp,
                        call=call,
                        response=response,
                        result=result,
                        event_type="subagent_event",
                    )
                else:
                    _map_evidence_tool(
                        events=events,
                        timestamp=timestamp,
                        call=call,
                        response=response,
                        result=result,
                        event_type="tool_result",
                    )

    patch_text = "\n\n".join(diffs) if diffs else None
    patch_hash = f"sha256:{hashlib.sha256(patch_text.encode('utf-8')).hexdigest()}" if patch_text else None
    tests_passed = None if last_test_exit_code is None else last_test_exit_code == 0
    final_text_failed = bool(final_response and _FAILURE_TEXT_RE.search(final_response))

    if error:
        status = "failed"
        escalation_reason = error
    elif completed is False and partial:
        status = "partial"
        escalation_reason = "Hermes trajectory marked partial."
    elif completed is False:
        status = "failed"
        escalation_reason = "Hermes trajectory marked incomplete."
    elif not events:
        status = "failed"
        escalation_reason = "no parseable Hermes trajectory events"
    elif had_command_failure or final_text_failed:
        status = "partial"
        if had_command_failure:
            escalation_reason = "one or more commands failed"
        else:
            escalation_reason = "final assistant response indicates failure"
    elif had_tool_error and not final_response:
        status = "partial"
        escalation_reason = "one or more tools returned errors"
    else:
        status = "completed"
        escalation_reason = None

    verification_summary = None
    if test_trace:
        verification_summary = "Latest test-like command passed." if tests_passed else "Latest test-like command failed."

    model = None
    if isinstance(metadata, dict):
        model = metadata.get("model") if isinstance(metadata.get("model"), str) else None

    outcome = EpisodeOutcome(
        status=status,
        tests_passed=tests_passed,
        verification_summary=verification_summary,
        escalation_reason=escalation_reason,
        files_touched=sorted(files_touched),
        final_response=final_response,
    )

    return HermesMappingResult(
        events=events,
        outcome=outcome,
        base_model_id=model,
        patch_text=patch_text,
        patch_hash=patch_hash,
        tool_trace=tool_trace or None,
        test_trace=test_trace or None,
        stdout_excerpt=_truncate(command_outputs[-1], 2000) if command_outputs else None,
        stderr_excerpt=_truncate(stderr_outputs[-1], 2000) if stderr_outputs else None,
        cost_tokens_prompt=None,
        cost_tokens_completion=None,
    )
