from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import Episode

from .item_mapper import map_hermes_conversations

DEFAULT_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class HermesMalformedLine:
    line_number: int
    text: str
    error: str


@dataclass
class HermesTrajectoryEntry:
    raw: dict
    conversations: list[dict]
    index: int
    source_path: str | None = None

    @property
    def prompt_index(self) -> int | None:
        value = self.raw.get("prompt_index")
        return int(value) if isinstance(value, int) else None

    @property
    def metadata(self) -> dict:
        value = self.raw.get("metadata")
        return value if isinstance(value, dict) else {}

    @property
    def completed(self) -> bool | None:
        value = self.raw.get("completed")
        return value if isinstance(value, bool) else None

    @property
    def partial(self) -> bool | None:
        value = self.raw.get("partial")
        return value if isinstance(value, bool) else None

    @property
    def error(self) -> str | None:
        value = self.raw.get("error")
        return str(value) if value else None


@dataclass
class HermesTrajectoryBatch:
    entries: list[HermesTrajectoryEntry]
    malformed_lines: list[HermesMalformedLine]
    source_path: str | None = None


def parse_hermes_datetime(value: object, fallback: datetime = DEFAULT_TIMESTAMP) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return fallback
    return fallback


def _entry_timestamp(entry: HermesTrajectoryEntry) -> datetime:
    metadata = entry.metadata
    return parse_hermes_datetime(
        entry.raw.get("timestamp") or metadata.get("timestamp"),
        fallback=DEFAULT_TIMESTAMP,
    )


def load_trajectory_lines(lines: Iterable[str], source_path: str | None = None) -> HermesTrajectoryBatch:
    entries: list[HermesTrajectoryEntry] = []
    malformed: list[HermesMalformedLine] = []

    for line_number, raw_line in enumerate(lines, start=1):
        text = raw_line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            malformed.append(HermesMalformedLine(line_number=line_number, text=text, error=str(exc)))
            continue
        if not isinstance(parsed, dict):
            malformed.append(HermesMalformedLine(line_number=line_number, text=text, error="line is not a JSON object"))
            continue
        conversations = parsed.get("conversations")
        if not isinstance(conversations, list) or not all(isinstance(item, dict) for item in conversations):
            malformed.append(
                HermesMalformedLine(
                    line_number=line_number,
                    text=text,
                    error="line does not contain a conversations array",
                )
            )
            continue
        entries.append(
            HermesTrajectoryEntry(
                raw=parsed,
                conversations=conversations,
                index=len(entries),
                source_path=source_path,
            )
        )

    return HermesTrajectoryBatch(entries=entries, malformed_lines=malformed, source_path=source_path)


def load_trajectory_jsonl(path: str | Path) -> HermesTrajectoryBatch:
    trajectory_path = Path(path)
    with trajectory_path.open("r", encoding="utf-8") as f:
        return load_trajectory_lines(f, source_path=str(trajectory_path))


def load_trajectory_json(path: str | Path) -> HermesTrajectoryBatch:
    """
    Load a pretty-printed sample JSON file produced by `run_agent.py --save-sample`.

    The batch runner writes JSONL; this helper exists for the adjacent Hermes
    sample format and still returns a batch wrapper for API consistency.
    """
    trajectory_path = Path(path)
    parsed = json.loads(trajectory_path.read_text(encoding="utf-8"))
    raw_entries = parsed if isinstance(parsed, list) else [parsed]
    lines = [json.dumps(entry, ensure_ascii=False) for entry in raw_entries if isinstance(entry, dict)]
    return load_trajectory_lines(lines, source_path=str(trajectory_path))


def _stable_missing_run_id(entry: HermesTrajectoryEntry) -> str:
    digest = hashlib.sha256(json.dumps(entry.raw, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"missing-run-{digest[:16]}"


def _entry_run_ref(entry: HermesTrajectoryEntry) -> str:
    metadata = entry.metadata
    batch_num = metadata.get("batch_num")
    prompt_index = entry.prompt_index
    if batch_num is not None and prompt_index is not None:
        return f"batch-{batch_num}:prompt-{prompt_index}"
    if prompt_index is not None:
        return f"prompt-{prompt_index}"
    return _stable_missing_run_id(entry)


def _stable_episode_id(run_id: str, task_id: str, mode: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes-agent:{run_id}:{task_id}:{mode}"))


def _first_human_prompt(conversations: list[dict]) -> str | None:
    for message in conversations:
        native_from = message.get("from") or message.get("role")
        if native_from in {"human", "user"} and isinstance(message.get("value"), str):
            return message["value"]
        if native_from in {"human", "user"} and isinstance(message.get("content"), str):
            return message["content"]
    return None


def trajectory_to_episode(
    entry: HermesTrajectoryEntry,
    *,
    task_id: str,
    task_domain: str,
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
    agent_surface: str = "hermes_agent",
) -> Episode:
    timestamp = _entry_timestamp(entry)
    mapping = map_hermes_conversations(
        entry.conversations,
        timestamp=timestamp,
        metadata=entry.metadata,
        completed=entry.completed,
        partial=entry.partial,
        error=entry.error,
    )
    # Hermes batch trajectories currently expose one timestamp per saved
    # trajectory entry, not per conversation message.
    started_at = timestamp
    ended_at = timestamp

    run_ref = _entry_run_ref(entry)
    run_id = f"hermes-agent:{run_ref}"
    thread_id = run_ref if not run_ref.startswith("missing-run-") else None
    description = task_description or _first_human_prompt(entry.conversations) or task_id

    return Episode(
        episode_id=_stable_episode_id(run_id, task_id, mode),
        run_id=run_id,
        thread_id=thread_id,
        task_id=task_id,
        task_description=description,
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


def batch_to_episodes(
    batch: HermesTrajectoryBatch,
    *,
    task_id_prefix: str,
    task_domain: str,
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
) -> list[Episode]:
    episodes: list[Episode] = []
    for entry in batch.entries:
        suffix = entry.prompt_index if entry.prompt_index is not None else entry.index
        episodes.append(
            trajectory_to_episode(
                entry,
                task_id=f"{task_id_prefix}-{suffix}",
                task_domain=task_domain,
                task_description=task_description,
                mode=mode,
            )
        )
    return episodes


def import_trajectory_jsonl(
    path: str | Path,
    *,
    task_id_prefix: str,
    task_domain: str,
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
) -> list[Episode]:
    return batch_to_episodes(
        load_trajectory_jsonl(path),
        task_id_prefix=task_id_prefix,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
    )


def append_trajectory_episodes(
    trajectory_path: str | Path,
    episodes_path: str | Path,
    *,
    task_id_prefix: str,
    task_domain: str,
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
) -> list[Episode]:
    episodes = import_trajectory_jsonl(
        trajectory_path,
        task_id_prefix=task_id_prefix,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
    )
    recorder = EpisodeRecorder(episodes_path)
    for episode in episodes:
        recorder.append(episode)
    return episodes
