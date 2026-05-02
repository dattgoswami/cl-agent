from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from cl_layer.episode.schema import EpisodeEvent, EpisodeOutcome

from .log_loader import AiderChatMessage

MAX_TEXT_CHARS = 4000
MAX_EXCERPT_CHARS = 2000

_FAILURE_TEXT_RE = re.compile(
    r"\b(failed|failure|error|aborted|blocked|could not|couldn't|cannot complete)\b",
    re.IGNORECASE,
)
_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|unittest|npm\s+(run\s+)?test|pnpm\s+test|yarn\s+test|"
    r"vitest|go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|npm\s+run\s+check|"
    r"ruff|flake8|mypy|eslint|tsc)\b"
)
_EXIT_CODE_RE = re.compile(r"(?:exit code|exited with code)\s*:?\s*(-?\d+)", re.IGNORECASE)
_RUNNING_RE = re.compile(r"^(?:#+\s*)?Running:\s*(.+?)$|^Running\s+(.+?)$", re.MULTILINE)


@dataclass(frozen=True)
class AiderMappingResult:
    events: list[EpisodeEvent]
    outcome: EpisodeOutcome
    patch_text: str | None
    patch_hash: str | None
    tool_trace: list[dict] | None
    test_trace: list[dict] | None
    stdout_excerpt: str | None
    stderr_excerpt: str | None
    latency_ms: float | None


def _truncate(text: str | None, limit: int = MAX_TEXT_CHARS) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _is_test_command(command: str | None) -> bool:
    return bool(command and _TEST_COMMAND_RE.search(command))


def _extract_exit_code(text: str | None) -> int | None:
    if not text:
        return None
    match = _EXIT_CODE_RE.search(text)
    return int(match.group(1)) if match else None


def _tool_commands(messages: Sequence[AiderChatMessage]) -> list[dict]:
    commands: list[dict] = []
    for message in messages:
        if message.role != "tool":
            continue
        matches = list(_RUNNING_RE.finditer(message.content))
        for index, match in enumerate(matches):
            command = (match.group(1) or match.group(2) or "").strip()
            if not command:
                continue
            slice_end = matches[index + 1].start() if index + 1 < len(matches) else len(message.content)
            slice_text = message.content[match.start():slice_end]
            commands.append(
                {
                    "command": command,
                    "output": _truncate(message.content),
                    "exit_code": _extract_exit_code(slice_text),
                    "message_index": message.index,
                    "source": message.source,
                }
            )
    return commands


def map_aider_run(
    *,
    args: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    cwd: str | None,
    started_at: datetime,
    ended_at: datetime,
    chat_messages: Sequence[AiderChatMessage] = (),
    changed_files: Sequence[str] = (),
    patch_text: str | None = None,
    preexisting_dirty_files: Sequence[str] = (),
    commit_before: str | None = None,
    commit_after: str | None = None,
) -> AiderMappingResult:
    events: list[EpisodeEvent] = []
    tool_trace: list[dict] = []
    test_trace: list[dict] = []
    files_touched = sorted(dict.fromkeys(str(path) for path in changed_files if path))
    patch_hash = f"sha256:{hashlib.sha256(patch_text.encode('utf-8')).hexdigest()}" if patch_text else None
    ts = started_at

    stdout_excerpt = _truncate(stdout, MAX_EXCERPT_CHARS)
    stderr_excerpt = _truncate(stderr, MAX_EXCERPT_CHARS)

    events.append(
        EpisodeEvent(
            kind="command_execution",
            timestamp=ts,
            payload={
                "source": "aider_subprocess",
                "args": list(args),
                "cwd": cwd,
                "exit_code": returncode,
                "stdout": stdout_excerpt,
                "stderr": stderr_excerpt,
            },
        )
    )
    tool_trace.append(
        {
            "source": "aider_subprocess",
            "args": list(args),
            "exit_code": returncode,
        }
    )

    for command in _tool_commands(chat_messages):
        exit_code = command["exit_code"]
        events.append(
            EpisodeEvent(
                kind="command_execution",
                timestamp=ended_at,
                payload={
                    "source": command["source"],
                    "message_index": command["message_index"],
                    "command": command["command"],
                    "exit_code": exit_code,
                    "output": command["output"],
                },
            )
        )
        tool_trace.append({"phase": "tool_output", **command})
        if _is_test_command(command["command"]) and exit_code is not None:
            test_trace.append(
                {
                    "command": command["command"],
                    "exit_code": exit_code,
                    "source": command["source"],
                }
            )

    if files_touched or patch_text:
        events.append(
            EpisodeEvent(
                kind="file_change",
                timestamp=ended_at,
                payload={
                    "source": "git_diff",
                    "paths": files_touched,
                    "patch_hash": patch_hash,
                    "patch_chars": len(patch_text) if patch_text else 0,
                    "commit_before": commit_before,
                    "commit_after": commit_after,
                    "preexisting_dirty_files": sorted(set(preexisting_dirty_files)),
                },
            )
        )

    final_response: str | None = None
    for message in chat_messages:
        if message.role == "assistant":
            final_response = message.content
        events.append(
            EpisodeEvent(
                kind="agent_message",
                timestamp=ended_at,
                payload={
                    "source": message.source,
                    "role": message.role,
                    "text": _truncate(message.content),
                    "message_index": message.index,
                },
            )
        )

    final_text_failed = bool(final_response and _FAILURE_TEXT_RE.search(final_response))
    stderr_failed = bool(
        final_response is None and stderr_excerpt and _FAILURE_TEXT_RE.search(stderr_excerpt)
    )
    tests_passed = None
    if test_trace:
        latest = test_trace[-1]
        exit_code = latest.get("exit_code")
        tests_passed = exit_code == 0 if isinstance(exit_code, int) else None

    if returncode != 0 and files_touched:
        status = "partial"
        escalation_reason = f"aider exited with code {returncode}"
    elif returncode != 0:
        status = "failed"
        escalation_reason = f"aider exited with code {returncode}"
    elif stderr_failed:
        status = "partial" if files_touched else "failed"
        escalation_reason = "stderr indicates failure"
    elif final_text_failed:
        status = "partial" if files_touched else "failed"
        escalation_reason = "final assistant response indicates failure"
    elif tests_passed is False:
        status = "partial" if files_touched else "failed"
        escalation_reason = "latest test-like command failed"
    else:
        status = "completed"
        escalation_reason = None

    verification_summary = None
    if test_trace:
        verification_summary = "Latest test-like command passed." if tests_passed else "Latest test-like command failed."

    latency_ms = (ended_at - started_at).total_seconds() * 1000

    return AiderMappingResult(
        events=events,
        outcome=EpisodeOutcome(
            status=status,
            tests_passed=tests_passed,
            verification_summary=verification_summary,
            escalation_reason=escalation_reason,
            files_touched=files_touched,
            final_response=final_response,
        ),
        patch_text=patch_text,
        patch_hash=patch_hash,
        tool_trace=tool_trace or None,
        test_trace=test_trace or None,
        stdout_excerpt=stdout_excerpt,
        stderr_excerpt=stderr_excerpt,
        latency_ms=latency_ms,
    )
