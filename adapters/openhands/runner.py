"""
OpenHands adapter runner and importer.

OpenHandsImporter
-----------------
High-level entry point for importing V1 conversation exports (zip files or
filesystem directories) into CL episodes.  No OpenHands runtime, Docker, or
network access is required.

OpenHandsLiveRunner (DEFERRED)
-------------------------------
Stub for future live execution against a running OpenHands app server.
Requires an injectable API client and is intentionally not implemented — use
OpenHandsImporter for V1 export files instead.

Live-runner implementation notes (for future work):
  - Use OpenHands' V1 app-conversation API surface.  The stable export
    download route is
    ``GET /api/v1/app-conversations/{conversation_id}/download``.
  - Write downloaded zip bytes to a temporary ``.zip`` file before calling
    :func:`load_conversation`, because the importer accepts filesystem paths.
  - For testing, inject a fake API client that returns pre-canned event JSON.
  - Do not call the live API in tests; skip with ``pytest.mark.skip``.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import Episode

from .context_builder import ContextBuilder
from .conversation_loader import OpenHandsConversation, load_conversation
from .item_mapper import OpenHandsMappingResult, _parse_timestamp, _task_from_meta, map_conversation


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _stable_episode_id(conversation_id: str, mode: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"openhands:{conversation_id}:{mode}"))


def _infer_conversation_id(conv: OpenHandsConversation, path: str | Path | None) -> str:
    cid = conv.meta.get("conversation_id") or conv.meta.get("id")
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    if path:
        return Path(path).stem
    return str(uuid.uuid4())


def _infer_timestamps(conv: OpenHandsConversation) -> tuple[datetime, datetime]:
    now = _now()
    started_at = None
    ended_at = None

    for key in ("created_at", "started_at"):
        value = conv.meta.get(key)
        if value:
            started_at = _parse_timestamp(value)
            break

    for key in ("ended_at", "updated_at", "finished_at"):
        value = conv.meta.get(key)
        if value:
            ended_at = _parse_timestamp(value)
            break

    # Fall back to first/last event timestamps
    if started_at is None and conv.events:
        started_at = _parse_timestamp(conv.events[0].get("timestamp"))
    if ended_at is None and conv.events:
        ended_at = _parse_timestamp(conv.events[-1].get("timestamp"))

    started_at = started_at or now
    ended_at = ended_at or started_at
    return started_at, ended_at


def conversation_to_episode(
    conv: OpenHandsConversation,
    *,
    task_id: str | None = None,
    task_domain: str = "software_engineering",
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
    agent_surface: str = "openhands",
) -> Episode:
    """Convert a loaded :class:`~conversation_loader.OpenHandsConversation` to
    a normalized :class:`~cl_layer.episode.schema.Episode`.

    Parameters
    ----------
    conv:
        Loaded conversation (from :func:`~conversation_loader.load_conversation`).
    task_id:
        Explicit task identifier.  When omitted the conversation_id from
        meta.json (or the source filename stem) is used.
    task_domain:
        Domain / category string (e.g. ``"python"``, ``"frontend"``).
    task_description:
        Human-readable task description.  Falls back to meta.json ``title``
        / ``initial_message`` fields, then to *task_id*.
    mode:
        ``"baseline"`` — no CL artifacts injected; ``"integrated"`` — CL
        artifacts were injected before the run.
    agent_surface:
        Override the agent surface label (default ``"openhands"``).
    """
    if mode not in {"baseline", "integrated"}:
        raise ValueError("mode must be 'baseline' or 'integrated'")

    conversation_id = _infer_conversation_id(conv, conv.source_path)
    inferred_task_id = task_id or conversation_id
    started_at, ended_at = _infer_timestamps(conv)

    mapping: OpenHandsMappingResult = map_conversation(conv)

    inferred_description = (
        task_description
        or _task_from_meta(conv.meta)
        or inferred_task_id
    )

    return Episode(
        episode_id=_stable_episode_id(conversation_id, mode),
        run_id=f"openhands:{conversation_id}",
        thread_id=conversation_id,
        task_id=inferred_task_id,
        task_description=inferred_description,
        task_domain=task_domain,
        agent_surface=agent_surface,
        mode=mode,
        started_at=started_at,
        ended_at=ended_at,
        events=mapping.events,
        outcome=mapping.outcome,
        reward=None,
        base_model_id=mapping.base_model_id,
        patch_text=mapping.patch_text,
        patch_hash=mapping.patch_hash,
        tool_trace=mapping.tool_trace,
        test_trace=mapping.test_trace,
        stdout_excerpt=mapping.stdout_excerpt,
        stderr_excerpt=mapping.stderr_excerpt,
        cost_tokens_prompt=mapping.cost_tokens_prompt,
        cost_tokens_completion=mapping.cost_tokens_completion,
    )


def import_conversation(
    path: str | Path,
    *,
    task_id: str | None = None,
    task_domain: str = "software_engineering",
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
) -> Episode:
    """Load a V1 conversation from a zip or directory and return an Episode."""
    conv = load_conversation(path)
    return conversation_to_episode(
        conv,
        task_id=task_id,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
    )


def append_conversation_episode(
    conversation_path: str | Path,
    episodes_path: str | Path,
    *,
    task_id: str | None = None,
    task_domain: str = "software_engineering",
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
) -> Episode:
    """Import a conversation and append the resulting Episode to a JSONL file."""
    episode = import_conversation(
        conversation_path,
        task_id=task_id,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
    )
    EpisodeRecorder(episodes_path).append(episode)
    return episode


# ---------------------------------------------------------------------------
# High-level importer class
# ---------------------------------------------------------------------------

class OpenHandsImporter:
    """Import OpenHands V1 conversation exports into CL episodes.

    Usage::

        importer = OpenHandsImporter(
            episodes_path="data/episodes.jsonl",
            artifacts_dir="data/",       # for baseline vs integrated checks
        )
        episode = importer.import_conversation(
            "exports/conv-abc.zip",
            task_id="task-001",
            task_domain="python",
            mode="baseline",
        )
    """

    def __init__(
        self,
        episodes_path: str | Path,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self.recorder = EpisodeRecorder(episodes_path)
        self.context_builder = ContextBuilder(artifacts_dir)

    def import_conversation(
        self,
        path: str | Path,
        *,
        task_id: str | None = None,
        task_domain: str = "software_engineering",
        task_description: str | None = None,
        mode: Literal["baseline", "integrated"] = "baseline",
    ) -> Episode:
        """Import a single conversation and append it to the recorder."""
        episode = import_conversation(
            path,
            task_id=task_id,
            task_domain=task_domain,
            task_description=task_description,
            mode=mode,
        )
        self.recorder.append(episode)
        return episode

    def import_conversations(
        self,
        paths: list[str | Path],
        *,
        task_domain: str = "software_engineering",
        mode: Literal["baseline", "integrated"] = "baseline",
    ) -> list[Episode]:
        """Import a list of conversations and append all to the recorder."""
        episodes: list[Episode] = []
        for path in paths:
            ep = import_conversation(path, task_domain=task_domain, mode=mode)
            self.recorder.append(ep)
            episodes.append(ep)
        return episodes


# ---------------------------------------------------------------------------
# Deferred live runner (stub)
# ---------------------------------------------------------------------------

class OpenHandsLiveRunner:
    """
    DEFERRED — live execution against a running OpenHands app server.

    This class is a documented stub.  The importer path is stable and
    recommended.  The live runner requires:

    - A running OpenHands app server (not started in tests).
    - An injectable API client (``api_client`` parameter).
    - Network access and model API keys.

    Tests must use ``pytest.mark.skip`` or inject a fake client.

    Implementation notes for follow-up work
    ----------------------------------------
    1. Use OpenHands' V1 app-conversation API surface to create or attach to
       a conversation; verify the start/event routes against the current
       OpenHands version before implementing.
    2. Poll or stream V1 events until execution reaches a terminal state.
    3. Fetch the full export from
       ``GET /api/v1/app-conversations/{conversation_id}/download``.
    4. Write the returned zip bytes to a temporary ``.zip`` file, then pass
       that path to :func:`load_conversation` and :func:`conversation_to_episode`.
    5. The ``api_client`` should be an injectable protocol / duck type so
       tests can replace it with a fake that returns pre-canned event JSON.
    """

    def __init__(self, api_client: object = None) -> None:
        if api_client is None:
            raise RuntimeError(
                "OpenHandsLiveRunner requires an injected api_client. "
                "Use OpenHandsImporter for V1 export files, or inject a "
                "fake API client in tests (skip live server calls)."
            )
        self._client = api_client

    def run(self, task_prompt: str, **kwargs: object) -> Episode:
        raise NotImplementedError(
            "Live runner not yet implemented. "
            "Use OpenHandsImporter to import V1 export files."
        )
