"""
Map Codex SDK ThreadItems into normalized CL-layer EpisodeEvents.

Imports cl_layer from the standard Python path — install cl-layer
(editable) or add the cl-layer directory to PYTHONPATH before use.
"""
from __future__ import annotations

from datetime import datetime, timezone

from cl_layer.episode.schema import EpisodeEvent, EpisodeOutcome


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def map_thread_items(
    items: list,
    final_response: str | None = None,
) -> tuple[list[EpisodeEvent], EpisodeOutcome]:
    """
    Convert a list of Codex ThreadItem objects into (events, outcome).

    Uses duck-typing on the `type` discriminator so the function works
    both with live Pydantic models and with lightweight test stubs.
    """
    events: list[EpisodeEvent] = []
    files_touched: list[str] = []
    had_command_failure = False
    ts = _now()

    for item in items:
        item_type = getattr(item, "type", None)

        if item_type == "commandExecution":
            exit_code = getattr(item, "exit_code", None)
            status = getattr(item, "status", None)
            events.append(
                EpisodeEvent(
                    kind="command_execution",
                    timestamp=ts,
                    payload={
                        "id": getattr(item, "id", None),
                        "command": getattr(item, "command", None),
                        "cwd": getattr(item, "cwd", None),
                        "exit_code": exit_code,
                        "status": str(status) if status is not None else None,
                        "duration_ms": getattr(item, "duration_ms", None),
                        "aggregated_output": getattr(item, "aggregated_output", None),
                    },
                )
            )
            if exit_code is not None and exit_code != 0:
                had_command_failure = True

        elif item_type == "fileChange":
            # FileChangeThreadItem.changes: list[FileUpdateChange]
            # FileUpdateChange has .path (str)
            changes = getattr(item, "changes", [])
            paths: list[str] = []
            for ch in changes:
                p = getattr(ch, "path", None) or (ch.get("path") if isinstance(ch, dict) else None)
                if p:
                    paths.append(str(p))
            files_touched.extend(paths)
            events.append(
                EpisodeEvent(
                    kind="file_change",
                    timestamp=ts,
                    payload={
                        "id": getattr(item, "id", None),
                        "status": str(getattr(item, "status", None)),
                        "paths": paths,
                    },
                )
            )

        elif item_type == "mcpToolCall":
            events.append(
                EpisodeEvent(
                    kind="mcp_tool_call",
                    timestamp=ts,
                    payload={
                        "id": getattr(item, "id", None),
                        "tool": getattr(item, "tool", None),
                        "server": getattr(item, "server", None),
                        "status": str(getattr(item, "status", None)),
                        "duration_ms": getattr(item, "duration_ms", None),
                    },
                )
            )

        elif item_type == "agentMessage":
            events.append(
                EpisodeEvent(
                    kind="agent_message",
                    timestamp=ts,
                    payload={
                        "id": getattr(item, "id", None),
                        "text": getattr(item, "text", None),
                        "phase": str(getattr(item, "phase", None)),
                    },
                )
            )
        # Other item types (reasoning, plan, hookPrompt, etc.) are ignored
        # — they carry no evidence relevant to the episode model today.

    # Derive a coarse outcome status from observable signals.
    if not events and final_response is None:
        outcome_status = "failed"
    elif had_command_failure:
        outcome_status = "partial"
    else:
        outcome_status = "completed"

    outcome = EpisodeOutcome(
        status=outcome_status,
        tests_passed=None,      # filled by evaluator
        verification_summary=None,
        escalation_reason=None,
        files_touched=sorted(set(files_touched)),
        final_response=final_response,
    )

    return events, outcome
