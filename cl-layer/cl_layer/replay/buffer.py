from __future__ import annotations

from dataclasses import dataclass

from ..episode.recorder import EpisodeRecorder
from ..episode.schema import Episode


@dataclass
class ReplayResult:
    failed_recent: list[Episode]
    success_same_domain: list[Episode]


class ReplayBuffer:
    """
    Keyword/domain-heuristic replay — no embeddings, no vector DB.

    Priority:
      1. recent failures (failed / partial / escalated) — useful as warnings
      2. recent successes in the same domain — useful as positive patterns
    """

    def __init__(self, recorder: EpisodeRecorder) -> None:
        self.recorder = recorder

    def query(
        self,
        domain: str | None = None,
        mode: str | None = None,
        max_failures: int = 5,
        max_successes: int = 5,
    ) -> ReplayResult:
        episodes = self.recorder.load_all()

        if mode is not None:
            episodes = [ep for ep in episodes if ep.mode == mode]

        # most-recent first
        by_recency = sorted(episodes, key=lambda ep: ep.ended_at, reverse=True)

        failed = [
            ep
            for ep in by_recency
            if ep.outcome.status in ("failed", "partial", "escalated")
        ][:max_failures]

        successes = [
            ep
            for ep in by_recency
            if ep.outcome.status == "completed"
            and (domain is None or ep.task_domain == domain)
        ][:max_successes]

        return ReplayResult(failed_recent=failed, success_same_domain=successes)

    def query_by_task(self, task_id: str) -> list[Episode]:
        return [ep for ep in self.recorder.load_all() if ep.task_id == task_id]

    def query_by_domain(self, domain: str) -> list[Episode]:
        return [ep for ep in self.recorder.load_all() if ep.task_domain == domain]
