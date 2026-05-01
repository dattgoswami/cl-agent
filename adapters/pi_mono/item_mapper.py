from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

from cl_layer.episode.schema import EpisodeEvent, EpisodeOutcome

from .time_utils import DEFAULT_TIMESTAMP, parse_pi_datetime

MAX_TEXT_CHARS = 4000
MAX_ARG_TEXT_CHARS = 500

_EXIT_CODE_RE = re.compile(r"Command exited with code\s+(-?\d+)")
_FAILURE_TEXT_RE = re.compile(
    r"\b(failed|failure|error|aborted|blocked|could not|couldn't|cannot complete)\b",
    re.IGNORECASE,
)
_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|unittest|npm\s+(run\s+)?test|pnpm\s+test|yarn\s+test|"
    r"vitest|go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|npm\s+run\s+check)\b"
)


@dataclass
class PiMappingResult:
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


def _entry_timestamp(entry: dict, fallback: datetime | None = None) -> datetime:
    message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
    parsed = parse_pi_datetime(entry.get("timestamp") or message.get("timestamp"), fallback or DEFAULT_TIMESTAMP)
    return parsed or DEFAULT_TIMESTAMP


def _truncate(text: str | None, limit: int = MAX_TEXT_CHARS) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _content_text(content: object, limit: int = MAX_TEXT_CHARS) -> tuple[str | None, int, int]:
    if isinstance(content, str):
        return _truncate(content, limit), 0, 0
    if not isinstance(content, list):
        return None, 0, 0

    parts: list[str] = []
    image_count = 0
    thinking_chars = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block_type == "thinking" and isinstance(block.get("thinking"), str):
            thinking_chars += len(block["thinking"])
        elif block_type == "image":
            image_count += 1
    text = "\n".join(part for part in parts if part)
    return _truncate(text or None, limit), image_count, thinking_chars


def _tool_calls(content: object) -> list[dict]:
    if not isinstance(content, list):
        return []
    calls: list[dict] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "toolCall":
            calls.append(
                {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("arguments") if isinstance(block.get("arguments"), dict) else {},
                }
            )
    return calls


def _path_from_args(args: dict | None) -> str | None:
    if not isinstance(args, dict):
        return None
    path = args.get("path") or args.get("file_path")
    return str(path) if path else None


def _compact_args(tool_name: str | None, args: dict | None) -> dict:
    if not isinstance(args, dict):
        return {}
    if tool_name == "bash":
        return {
            "command": _truncate(str(args.get("command")), MAX_ARG_TEXT_CHARS) if args.get("command") is not None else None,
            "timeout": args.get("timeout"),
        }
    if tool_name == "edit":
        edits = args.get("edits")
        return {
            "path": _path_from_args(args),
            "edits_count": len(edits) if isinstance(edits, list) else None,
            "has_legacy_replacement": isinstance(args.get("oldText"), str) or isinstance(args.get("newText"), str),
        }
    if tool_name == "write":
        content = args.get("content")
        return {
            "path": _path_from_args(args),
            "content_chars": len(content) if isinstance(content, str) else None,
        }
    if tool_name == "read":
        return {
            "path": _path_from_args(args),
            "offset": args.get("offset"),
            "limit": args.get("limit"),
        }
    compact: dict[str, object] = {}
    for key, value in args.items():
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


def _text_from_tool_result(message: dict) -> str | None:
    text, _, _ = _content_text(message.get("content"))
    return text


def _extract_exit_code(text: str | None, is_error: bool | None = None) -> int | None:
    if not text:
        return 1 if is_error else None
    match = _EXIT_CODE_RE.search(text)
    if match:
        return int(match.group(1))
    return 1 if is_error else None


def _is_test_command(command: str | None) -> bool:
    return bool(command and _TEST_COMMAND_RE.search(command))


def _usage_tokens(usage: object) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("input")
    completion = usage.get("output")
    return (
        int(prompt) if isinstance(prompt, (int, float)) else None,
        int(completion) if isinstance(completion, (int, float)) else None,
    )


def _append_agent_message(
    events: list[EpisodeEvent],
    timestamp: datetime,
    entry: dict,
    payload: dict,
) -> None:
    payload.setdefault("entry_id", entry.get("id"))
    payload.setdefault("parent_id", entry.get("parentId"))
    payload.setdefault("entry_type", entry.get("type"))
    events.append(EpisodeEvent(kind="agent_message", timestamp=timestamp, payload=payload))


def map_pi_entries(entries: list[dict], header: dict | None = None) -> PiMappingResult:
    events: list[EpisodeEvent] = []
    files_touched: set[str] = set()
    diffs: list[str] = []
    tool_trace: list[dict] = []
    test_trace: list[dict] = []
    command_outputs: list[str] = []

    pending_tool_calls: dict[str, dict] = {}
    explicit_failure: str | None = None
    explicit_abort = False
    had_command_failure = False
    last_test_exit_code: int | None = None
    final_response: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_tokens = 0
    completion_tokens = 0
    saw_prompt_tokens = False
    saw_completion_tokens = False

    if isinstance(header, dict):
        provider = header.get("provider") if isinstance(header.get("provider"), str) else None
        model = (
            header.get("modelId")
            if isinstance(header.get("modelId"), str)
            else header.get("model")
            if isinstance(header.get("model"), str)
            else None
        )

    for entry in entries:
        timestamp = _entry_timestamp(entry, events[-1].timestamp if events else None)
        entry_type = entry.get("type")

        if entry_type == "message" and isinstance(entry.get("message"), dict):
            message = entry["message"]
            role = message.get("role")

            if role == "bashExecution":
                output = _truncate(message.get("output") if isinstance(message.get("output"), str) else None)
                exit_code = message.get("exitCode")
                cancelled = bool(message.get("cancelled"))
                failed = cancelled or (isinstance(exit_code, int) and exit_code != 0)
                command = message.get("command") if isinstance(message.get("command"), str) else None
                if failed:
                    had_command_failure = True
                if cancelled:
                    explicit_abort = True
                if _is_test_command(command) and isinstance(exit_code, int):
                    last_test_exit_code = exit_code
                    test_trace.append({"command": command, "exit_code": exit_code, "source": "bashExecution"})
                if output:
                    command_outputs.append(output)

                events.append(
                    EpisodeEvent(
                        kind="command_execution",
                        timestamp=timestamp,
                        payload={
                            "entry_id": entry.get("id"),
                            "parent_id": entry.get("parentId"),
                            "source": "bashExecution",
                            "command": command,
                            "exit_code": exit_code,
                            "cancelled": cancelled,
                            "truncated": bool(message.get("truncated")),
                            "full_output_path": message.get("fullOutputPath"),
                            "exclude_from_context": message.get("excludeFromContext"),
                            "output": output,
                        },
                    )
                )
                continue

            if role == "assistant":
                provider = message.get("provider") if isinstance(message.get("provider"), str) else provider
                model = message.get("model") if isinstance(message.get("model"), str) else model
                usage_prompt, usage_completion = _usage_tokens(message.get("usage"))
                if usage_prompt is not None:
                    prompt_tokens += usage_prompt
                    saw_prompt_tokens = True
                if usage_completion is not None:
                    completion_tokens += usage_completion
                    saw_completion_tokens = True

                text, image_count, thinking_chars = _content_text(message.get("content"))
                if text:
                    final_response = text
                calls = _tool_calls(message.get("content"))
                compact_calls: list[dict] = []
                for call in calls:
                    call_id = call.get("id")
                    tool_name = call.get("name")
                    args = call.get("arguments")
                    if isinstance(call_id, str):
                        pending_tool_calls[call_id] = {
                            "id": call_id,
                            "name": tool_name,
                            "arguments": args,
                            "entry_id": entry.get("id"),
                            "timestamp": timestamp.isoformat(),
                        }
                    compact = {"id": call_id, "name": tool_name, "arguments": _compact_args(tool_name, args)}
                    compact_calls.append(compact)
                    tool_trace.append({"phase": "call", **compact})

                stop_reason = message.get("stopReason")
                error_message = message.get("errorMessage")
                if stop_reason == "error":
                    explicit_failure = str(error_message or "assistant stopReason=error")
                elif stop_reason == "aborted":
                    explicit_abort = True
                    explicit_failure = str(error_message or "assistant stopReason=aborted")

                _append_agent_message(
                    events,
                    timestamp,
                    entry,
                    {
                        "role": "assistant",
                        "text": text,
                        "image_count": image_count,
                        "thinking_chars": thinking_chars,
                        "provider": provider,
                        "model": model,
                        "stop_reason": stop_reason,
                        "error_message": error_message,
                        "tool_calls": compact_calls,
                    },
                )
                continue

            if role == "toolResult":
                tool_call_id = message.get("toolCallId")
                tool_name = message.get("toolName")
                call = pending_tool_calls.get(tool_call_id) if isinstance(tool_call_id, str) else None
                if not tool_name and call:
                    tool_name = call.get("name")
                args = call.get("arguments") if call else None
                result_text = _text_from_tool_result(message)
                is_error = bool(message.get("isError"))
                details = message.get("details") if isinstance(message.get("details"), dict) else None
                tool_trace.append(
                    {
                        "phase": "result",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "is_error": is_error,
                        "text": _truncate(result_text, MAX_ARG_TEXT_CHARS),
                        "details_keys": sorted(details.keys()) if details else [],
                    }
                )

                if tool_name == "bash":
                    command = args.get("command") if isinstance(args, dict) and isinstance(args.get("command"), str) else None
                    exit_code = _extract_exit_code(result_text, is_error)
                    if is_error or (exit_code is not None and exit_code != 0):
                        had_command_failure = True
                    if _is_test_command(command) and exit_code is not None:
                        last_test_exit_code = exit_code
                        test_trace.append({"command": command, "exit_code": exit_code, "source": "toolResult"})
                    if result_text:
                        command_outputs.append(result_text)

                    events.append(
                        EpisodeEvent(
                            kind="command_execution",
                            timestamp=timestamp,
                            payload={
                                "entry_id": entry.get("id"),
                                "parent_id": entry.get("parentId"),
                                "source": "toolResult",
                                "tool_call_id": tool_call_id,
                                "command": command,
                                "exit_code": exit_code,
                                "is_error": is_error,
                                "output": result_text,
                                "details": details,
                            },
                        )
                    )
                    continue

                if tool_name in {"edit", "write", "read"}:
                    path = _path_from_args(args)
                    if path:
                        mutating = tool_name in {"edit", "write"}
                        if mutating and not is_error:
                            files_touched.add(path)
                        diff = details.get("diff") if details and isinstance(details.get("diff"), str) else None
                        if diff:
                            diffs.append(diff)
                        events.append(
                            EpisodeEvent(
                                kind="file_change",
                                timestamp=timestamp,
                                payload={
                                    "entry_id": entry.get("id"),
                                    "parent_id": entry.get("parentId"),
                                    "source": "toolResult",
                                    "tool_call_id": tool_call_id,
                                    "operation": tool_name,
                                    "path": path,
                                    "paths": [path],
                                    "mutating": mutating,
                                    "is_error": is_error,
                                    "result": result_text,
                                    "diff": diff,
                                    "details": details,
                                },
                            )
                        )
                        continue

                _append_agent_message(
                    events,
                    timestamp,
                    entry,
                    {
                        "role": "toolResult",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "is_error": is_error,
                        "text": result_text,
                    },
                )
                continue

            if role in {"user", "custom"}:
                text, image_count, thinking_chars = _content_text(message.get("content"))
                payload = {
                    "role": role,
                    "text": text,
                    "image_count": image_count,
                    "thinking_chars": thinking_chars,
                }
                if role == "custom":
                    payload.update(
                        {
                            "custom_type": message.get("customType"),
                            "display": message.get("display"),
                            "details": message.get("details"),
                        }
                    )
                _append_agent_message(events, timestamp, entry, payload)
                continue

            if role in {"branchSummary", "compactionSummary"}:
                _append_agent_message(
                    events,
                    timestamp,
                    entry,
                    {
                        "role": role,
                        "summary": _truncate(message.get("summary") if isinstance(message.get("summary"), str) else None),
                        "from_id": message.get("fromId"),
                        "tokens_before": message.get("tokensBefore"),
                    },
                )
                continue

        elif entry_type == "branch_summary":
            _append_agent_message(
                events,
                timestamp,
                entry,
                {
                    "role": "branchSummary",
                    "summary": _truncate(entry.get("summary") if isinstance(entry.get("summary"), str) else None),
                    "from_id": entry.get("fromId"),
                    "details": entry.get("details"),
                    "from_hook": entry.get("fromHook"),
                },
            )
            continue

        elif entry_type == "compaction":
            _append_agent_message(
                events,
                timestamp,
                entry,
                {
                    "role": "compactionSummary",
                    "summary": _truncate(entry.get("summary") if isinstance(entry.get("summary"), str) else None),
                    "first_kept_entry_id": entry.get("firstKeptEntryId"),
                    "tokens_before": entry.get("tokensBefore"),
                    "details": entry.get("details"),
                    "from_hook": entry.get("fromHook"),
                },
            )
            continue

        elif entry_type == "custom_message":
            text, image_count, _thinking_chars = _content_text(entry.get("content"))
            _append_agent_message(
                events,
                timestamp,
                entry,
                {
                    "role": "custom",
                    "custom_type": entry.get("customType"),
                    "text": text,
                    "image_count": image_count,
                    "display": entry.get("display"),
                    "details": entry.get("details"),
                },
            )
            continue

        elif entry_type == "model_change":
            provider = entry.get("provider") if isinstance(entry.get("provider"), str) else provider
            model = entry.get("modelId") if isinstance(entry.get("modelId"), str) else model
            _append_agent_message(
                events,
                timestamp,
                entry,
                {"role": "model_change", "provider": provider, "model": model},
            )
            continue

        elif entry_type == "thinking_level_change":
            _append_agent_message(
                events,
                timestamp,
                entry,
                {"role": "thinking_level_change", "thinking_level": entry.get("thinkingLevel")},
            )
            continue

    patch_text = "\n\n".join(diffs) if diffs else None
    patch_hash = f"sha256:{hashlib.sha256(patch_text.encode('utf-8')).hexdigest()}" if patch_text else None

    tests_passed = None if last_test_exit_code is None else last_test_exit_code == 0
    final_text_failed = bool(final_response and _FAILURE_TEXT_RE.search(final_response))

    if explicit_failure:
        status = "failed"
        escalation_reason = explicit_failure
    elif not events:
        status = "failed"
        escalation_reason = "no parseable Pi session events"
    elif explicit_abort:
        status = "failed"
        escalation_reason = "Pi session was aborted"
    elif had_command_failure or final_text_failed:
        status = "partial"
        escalation_reason = "one or more commands failed" if had_command_failure else None
    else:
        status = "completed"
        escalation_reason = None

    verification_summary = None
    if test_trace:
        verification_summary = "Latest test-like command passed." if tests_passed else "Latest test-like command failed."

    outcome = EpisodeOutcome(
        status=status,
        tests_passed=tests_passed,
        verification_summary=verification_summary,
        escalation_reason=escalation_reason,
        files_touched=sorted(files_touched),
        final_response=final_response,
    )

    base_model_id = f"{provider}/{model}" if provider and model else model
    stdout_excerpt = _truncate(command_outputs[-1], 2000) if command_outputs else None

    return PiMappingResult(
        events=events,
        outcome=outcome,
        base_model_id=base_model_id,
        patch_text=patch_text,
        patch_hash=patch_hash,
        tool_trace=tool_trace or None,
        test_trace=test_trace or None,
        stdout_excerpt=stdout_excerpt,
        stderr_excerpt=None,
        cost_tokens_prompt=prompt_tokens if saw_prompt_tokens else None,
        cost_tokens_completion=completion_tokens if saw_completion_tokens else None,
    )
