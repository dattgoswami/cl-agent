from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

EventKind = Literal[
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "agent_message",
    "evaluation_result",
]

EpisodeStatus = Literal["completed", "partial", "failed", "escalated"]

RunMode = Literal["baseline", "integrated"]


@dataclass
class EpisodeEvent:
    kind: EventKind
    timestamp: datetime
    payload: dict


@dataclass
class EpisodeOutcome:
    status: EpisodeStatus
    tests_passed: bool | None
    verification_summary: str | None
    escalation_reason: str | None
    files_touched: list[str]
    final_response: str | None


@dataclass
class Episode:
    episode_id: str
    run_id: str
    thread_id: str | None
    task_id: str
    task_description: str
    task_domain: str
    agent_surface: str
    mode: RunMode
    started_at: datetime
    ended_at: datetime
    events: list[EpisodeEvent]
    outcome: EpisodeOutcome
    reward: float | None = None


def new_episode_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize(v: object) -> object:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _serialize(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize(item) for item in v]
    return v


def episode_to_dict(ep: Episode) -> dict:
    return _serialize(asdict(ep))  # type: ignore[arg-type]


def episode_from_dict(d: dict) -> Episode:
    events = [
        EpisodeEvent(
            kind=ev["kind"],
            timestamp=datetime.fromisoformat(ev["timestamp"]),
            payload=ev["payload"],
        )
        for ev in d.get("events", [])
    ]
    od = d["outcome"]
    outcome = EpisodeOutcome(
        status=od["status"],
        tests_passed=od.get("tests_passed"),
        verification_summary=od.get("verification_summary"),
        escalation_reason=od.get("escalation_reason"),
        files_touched=od.get("files_touched", []),
        final_response=od.get("final_response"),
    )
    return Episode(
        episode_id=d["episode_id"],
        run_id=d["run_id"],
        thread_id=d.get("thread_id"),
        task_id=d["task_id"],
        task_description=d["task_description"],
        task_domain=d["task_domain"],
        agent_surface=d["agent_surface"],
        mode=d["mode"],
        started_at=datetime.fromisoformat(d["started_at"]),
        ended_at=datetime.fromisoformat(d["ended_at"]),
        events=events,
        outcome=outcome,
        reward=d.get("reward"),
    )
