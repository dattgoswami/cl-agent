from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from ..episode.schema import Episode


@dataclass
class SessionSummary:
    run_ids: list[str]
    attempted: int
    completed: int
    failed: int
    escalated: int
    domains: list[str]
    recurring_failure_domains: list[str]
    recommended_replay_ids: list[str]
    generated_at: datetime


def summarize_session(
    episodes: list[Episode],
    run_ids: list[str] | None = None,
) -> SessionSummary:
    eps = (
        [ep for ep in episodes if ep.run_id in run_ids]
        if run_ids is not None
        else list(episodes)
    )

    completed = [ep for ep in eps if ep.outcome.status == "completed"]
    failed = [ep for ep in eps if ep.outcome.status in ("failed", "partial")]
    escalated = [ep for ep in eps if ep.outcome.status == "escalated"]

    domain_fail_counts: Counter[str] = Counter(ep.task_domain for ep in failed + escalated)
    recurring = [d for d, count in domain_fail_counts.items() if count >= 2]

    replay_ids = [
        ep.episode_id
        for ep in sorted(failed + escalated, key=lambda e: e.ended_at, reverse=True)[:5]
    ]

    return SessionSummary(
        run_ids=run_ids or list(dict.fromkeys(ep.run_id for ep in eps)),
        attempted=len(eps),
        completed=len(completed),
        failed=len(failed),
        escalated=len(escalated),
        domains=list(dict.fromkeys(ep.task_domain for ep in eps)),
        recurring_failure_domains=recurring,
        recommended_replay_ids=replay_ids,
        generated_at=datetime.utcnow(),
    )


def render_dreams_md(summary: SessionSummary) -> str:
    ts = summary.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# DREAMS\n\n",
        f"_Session summary generated {ts}_\n\n",
        "## This Session\n\n",
        f"- Attempted: {summary.attempted}\n",
        f"- Completed: {summary.completed}\n",
        f"- Failed: {summary.failed}\n",
        f"- Escalated: {summary.escalated}\n\n",
        "## Domains\n\n",
        *[f"- {d}\n" for d in summary.domains],
    ]
    if summary.recurring_failure_domains:
        lines += ["\n## Recurring Failures\n\n"]
        lines += [f"- {d}\n" for d in summary.recurring_failure_domains]
    if summary.recommended_replay_ids:
        lines += ["\n## Recommended Replay Targets\n\n"]
        lines += [f"- {eid}\n" for eid in summary.recommended_replay_ids]
    return "".join(lines)
