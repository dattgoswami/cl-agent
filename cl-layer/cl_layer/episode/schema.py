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
    # Search / SOAR loop events
    "plan_generated",
    "patch_proposed",
    "patch_applied",
    "verification_started",
    "verification_finished",
    "repair_prompt_generated",
    "repair_candidate_generated",
    "benchmark_result",
    "human_feedback",
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

    # --- Training-grade fields (all optional, default None) ---
    repo_id: str | None = None
    repo_path: str | None = None
    git_commit: str | None = None
    base_model_id: str | None = None
    student_model_id: str | None = None
    parent_episode_id: str | None = None
    benchmark_split: str | None = None
    task_tags: list[str] | None = None

    verification_steps: list[dict] | None = None
    verification_score: float | None = None
    verification_failures: list[str] | None = None

    patch_text: str | None = None
    patch_hash: str | None = None

    tool_trace: list[dict] | None = None
    test_trace: list[dict] | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None

    cost_tokens_prompt: int | None = None
    cost_tokens_completion: int | None = None
    latency_ms: float | None = None

    candidate_rank: int | None = None
    population_id: str | None = None
    generation_id: str | None = None

    accepted_for_training: bool | None = None
    accepted_reason: str | None = None


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
        # Training-grade fields (all optional, backward-compatible)
        repo_id=d.get("repo_id"),
        repo_path=d.get("repo_path"),
        git_commit=d.get("git_commit"),
        base_model_id=d.get("base_model_id"),
        student_model_id=d.get("student_model_id"),
        parent_episode_id=d.get("parent_episode_id"),
        benchmark_split=d.get("benchmark_split"),
        task_tags=d.get("task_tags"),
        verification_steps=d.get("verification_steps"),
        verification_score=d.get("verification_score"),
        verification_failures=d.get("verification_failures"),
        patch_text=d.get("patch_text"),
        patch_hash=d.get("patch_hash"),
        tool_trace=d.get("tool_trace"),
        test_trace=d.get("test_trace"),
        stdout_excerpt=d.get("stdout_excerpt"),
        stderr_excerpt=d.get("stderr_excerpt"),
        cost_tokens_prompt=d.get("cost_tokens_prompt"),
        cost_tokens_completion=d.get("cost_tokens_completion"),
        latency_ms=d.get("latency_ms"),
        candidate_rank=d.get("candidate_rank"),
        population_id=d.get("population_id"),
        generation_id=d.get("generation_id"),
        accepted_for_training=d.get("accepted_for_training"),
        accepted_reason=d.get("accepted_reason"),
    )
