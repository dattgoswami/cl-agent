"""
Codex SDK runner — Phase 3 adapter.

Runs a single coding task through the Codex Python SDK / app-server,
maps the structured ThreadItems into a normalized substrate Episode,
and appends it to episodes.jsonl.

Usage:
    from adapters.codex.sdk_runner import CodexRunner

    runner = CodexRunner(
        episodes_path="data/episodes.jsonl",
        artifacts_dir="data/",       # reads PROGRAM.md / SKILLS.md from here
        default_model="o4-mini",
    )
    episode = runner.run(
        task_prompt="Add a /healthz endpoint to main.py",
        task_id="task-001",
        task_domain="fastapi",
        mode="baseline",             # or "integrated"
        cwd="/path/to/repo",
    )

Prerequisites:
    - Codex app-server running locally (or accessible)
    - cl-layer installed: `pip install -e path/to/cl-layer`
    - codex_app_server installed: `pip install -e path/to/codex/sdk/python`
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import Episode, new_episode_id

from .context_builder import ContextBuilder, RunContext
from .item_mapper import map_thread_items


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class CodexRunner:
    def __init__(
        self,
        episodes_path: str | Path,
        artifacts_dir: str | Path | None = None,
        default_model: str | None = None,
    ) -> None:
        self.recorder = EpisodeRecorder(episodes_path)
        self.context_builder = ContextBuilder(artifacts_dir)
        self.default_model = default_model

    def run(
        self,
        task_prompt: str,
        task_id: str,
        task_domain: str,
        mode: Literal["baseline", "integrated"] = "baseline",
        cwd: str | None = None,
        task_description: str | None = None,
        agent_surface: str = "codex",
    ) -> Episode:
        """
        Start a Codex thread, run the task, map items, persist episode.

        In `baseline` mode the thread is ephemeral so Codex native memory
        does not carry over between runs — the substrate is the only
        deliberate cross-session learning mechanism.

        In `integrated` mode Codex native memory is allowed to persist;
        the substrate adds structured logging and evaluation on top.
        """
        from codex_app_server import Codex

        run_id = str(uuid.uuid4())
        episode_id = new_episode_id()
        started_at = _now()

        ctx: RunContext = self.context_builder.build(
            task_prompt=task_prompt,
            mode=mode,
            cwd=cwd,
        )

        # baseline -> ephemeral thread to minimize native-memory carryover
        ephemeral = mode == "baseline"

        with Codex() as codex:
            thread = codex.thread_start(
                cwd=ctx.cwd,
                developer_instructions=ctx.developer_instructions,
                ephemeral=ephemeral,
                model=self.default_model,
            )
            result = thread.run(ctx.task_prompt)

        ended_at = _now()

        items = result.items or []
        final_response = result.final_response

        events, outcome = map_thread_items(items, final_response=final_response)

        # Thread.id is the canonical identifier (see codex/sdk/python/src/codex_app_server/api.py)
        thread_id = getattr(thread, "id", None)

        episode = Episode(
            episode_id=episode_id,
            run_id=run_id,
            thread_id=str(thread_id) if thread_id is not None else None,
            task_id=task_id,
            task_description=task_description or task_prompt[:200],
            task_domain=task_domain,
            agent_surface=agent_surface,
            mode=mode,
            started_at=started_at,
            ended_at=ended_at,
            events=events,
            outcome=outcome,
            reward=None,
        )

        self.recorder.append(episode)
        return episode
