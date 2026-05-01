from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import Episode

from .item_mapper import map_pi_entries
from .time_utils import DEFAULT_TIMESTAMP, parse_pi_datetime


@dataclass(frozen=True)
class PiMalformedLine:
    line_number: int
    text: str
    error: str


@dataclass
class PiSession:
    header: dict | None
    entries: list[dict]
    malformed_lines: list[PiMalformedLine]
    source_path: str | None = None

    @property
    def session_id(self) -> str | None:
        if isinstance(self.header, dict) and isinstance(self.header.get("id"), str):
            return self.header["id"]
        return None

    @property
    def cwd(self) -> str | None:
        if isinstance(self.header, dict) and isinstance(self.header.get("cwd"), str):
            return self.header["cwd"]
        return None


def load_session_lines(lines: Iterable[str], source_path: str | None = None) -> PiSession:
    header: dict | None = None
    entries: list[dict] = []
    malformed: list[PiMalformedLine] = []

    for line_number, raw_line in enumerate(lines, start=1):
        text = raw_line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            malformed.append(PiMalformedLine(line_number=line_number, text=text, error=str(exc)))
            continue
        if not isinstance(parsed, dict):
            malformed.append(PiMalformedLine(line_number=line_number, text=text, error="line is not a JSON object"))
            continue
        if parsed.get("type") == "session" and header is None:
            header = parsed
        else:
            entries.append(parsed)

    return PiSession(header=header, entries=entries, malformed_lines=malformed, source_path=source_path)


def load_session_jsonl(path: str | Path) -> PiSession:
    session_path = Path(path)
    with session_path.open("r", encoding="utf-8") as f:
        return load_session_lines(f, source_path=str(session_path))


def _entries_have_tree_ids(entries: list[dict]) -> bool:
    message_entries = [entry for entry in entries if entry.get("type") == "message"]
    return bool(message_entries) and all(isinstance(entry.get("id"), str) for entry in message_entries)


def select_branch_entries(entries: list[dict], leaf_id: str | None = None) -> list[dict]:
    """
    Select the Pi active branch path.

    Pi reconstructs the current leaf as the last appended entry when opening a
    session file.  If entries have v2/v3 tree ids, mirror that behavior and walk
    parent links from `leaf_id` or the last entry.  Legacy linear sessions
    without ids are returned in file order.
    """
    if not entries:
        return []
    if not _entries_have_tree_ids(entries):
        if leaf_id is not None:
            raise ValueError("Cannot select a branch by id from a legacy Pi session without entry ids")
        return list(entries)

    by_id = {entry["id"]: entry for entry in entries if isinstance(entry.get("id"), str)}
    last_id = next((entry["id"] for entry in reversed(entries) if isinstance(entry.get("id"), str)), None)
    selected_leaf_id = leaf_id or last_id
    if selected_leaf_id is None:
        return list(entries)
    if selected_leaf_id not in by_id:
        raise ValueError(f"Pi session entry not found: {selected_leaf_id}")

    branch: list[dict] = []
    seen: set[str] = set()
    current = by_id[selected_leaf_id]
    while current:
        current_id = current.get("id")
        if not isinstance(current_id, str) or current_id in seen:
            break
        seen.add(current_id)
        branch.append(current)
        parent_id = current.get("parentId")
        current = by_id.get(parent_id) if isinstance(parent_id, str) else None
    branch.reverse()
    return branch


def _stable_missing_session_id(session: PiSession) -> str:
    digest = hashlib.sha256()
    if session.header is not None:
        digest.update(json.dumps(session.header, sort_keys=True).encode("utf-8"))
    for entry in session.entries:
        digest.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
    return f"missing-session-{digest.hexdigest()[:16]}"


def _stable_episode_id(
    session_id: str,
    leaf_id: str | None,
    task_id: str,
    mode: str,
) -> str:
    key = f"pi-mono:{session_id}:{leaf_id or 'linear'}:{task_id}:{mode}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _event_bounds(header: dict | None, events: list) -> tuple[datetime, datetime]:
    header_time = parse_pi_datetime(header.get("timestamp") if isinstance(header, dict) else None, fallback=None)
    timestamps = [event.timestamp for event in events]
    candidates = ([header_time] if header_time is not None else []) + timestamps
    if not candidates:
        return DEFAULT_TIMESTAMP, DEFAULT_TIMESTAMP
    return min(candidates), max(candidates)


def _parent_metadata(header: dict | None) -> str | None:
    if not isinstance(header, dict):
        return None
    for key in ("parentSession", "branchedFrom", "parent_session", "parent"):
        value = header.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def session_to_episode(
    session: PiSession,
    *,
    task_id: str,
    task_domain: str,
    task_description: str,
    mode: Literal["baseline", "integrated"] = "baseline",
    branch_leaf_id: str | None = None,
    agent_surface: str = "pi_mono",
) -> Episode:
    branch_entries = select_branch_entries(session.entries, branch_leaf_id)
    mapping = map_pi_entries(branch_entries, header=session.header)
    session_id = session.session_id or _stable_missing_session_id(session)
    leaf_id = branch_leaf_id
    if leaf_id is None and branch_entries and isinstance(branch_entries[-1].get("id"), str):
        leaf_id = branch_entries[-1]["id"]

    started_at, ended_at = _event_bounds(session.header, mapping.events)
    parent_ref = _parent_metadata(session.header)

    return Episode(
        episode_id=_stable_episode_id(session_id, leaf_id, task_id, mode),
        run_id=f"pi-mono:{session_id}:{leaf_id or 'linear'}",
        thread_id=session_id,
        task_id=task_id,
        task_description=task_description,
        task_domain=task_domain,
        agent_surface=agent_surface,
        mode=mode,
        started_at=started_at,
        ended_at=ended_at,
        events=mapping.events,
        outcome=mapping.outcome,
        reward=None,
        repo_path=session.cwd,
        base_model_id=mapping.base_model_id,
        parent_episode_id=parent_ref,
        patch_text=mapping.patch_text,
        patch_hash=mapping.patch_hash,
        tool_trace=mapping.tool_trace,
        test_trace=mapping.test_trace,
        stdout_excerpt=mapping.stdout_excerpt,
        stderr_excerpt=mapping.stderr_excerpt,
        cost_tokens_prompt=mapping.cost_tokens_prompt,
        cost_tokens_completion=mapping.cost_tokens_completion,
    )


def import_session_jsonl(
    path: str | Path,
    *,
    task_id: str,
    task_domain: str,
    task_description: str,
    mode: Literal["baseline", "integrated"] = "baseline",
    branch_leaf_id: str | None = None,
) -> Episode:
    session = load_session_jsonl(path)
    return session_to_episode(
        session,
        task_id=task_id,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
        branch_leaf_id=branch_leaf_id,
    )


def append_session_episode(
    session_path: str | Path,
    episodes_path: str | Path,
    *,
    task_id: str,
    task_domain: str,
    task_description: str,
    mode: Literal["baseline", "integrated"] = "baseline",
    branch_leaf_id: str | None = None,
) -> Episode:
    episode = import_session_jsonl(
        session_path,
        task_id=task_id,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
        branch_leaf_id=branch_leaf_id,
    )
    EpisodeRecorder(episodes_path).append(episode)
    return episode
