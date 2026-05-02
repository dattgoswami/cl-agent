from __future__ import annotations

import json
import re
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import Episode

from .item_mapper import map_swe_agent_trajectory

DEFAULT_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)

_CONFIG_CANDIDATES = (
    "{stem}.config.yaml",
    "{stem}.config.yml",
    "config.yaml",
    "config.yml",
    "args.yaml",
    "args.yml",
    "run_batch.config.yaml",
    "run_batch.config.yml",
)


@dataclass(frozen=True)
class SWEAgentTrajectory:
    raw: dict
    trajectory: list[dict]
    history: list[dict]
    info: dict
    environment: object
    source_path: str | None = None
    source_config: str | dict | None = None
    source_config_path: str | None = None


def parse_swe_agent_datetime(value: object, fallback: datetime = DEFAULT_TIMESTAMP) -> datetime:
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


def _load_sibling_config(path: Path) -> tuple[str | None, str | None]:
    for pattern in _CONFIG_CANDIDATES:
        candidate = path.parent / pattern.format(stem=path.stem)
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8"), str(candidate)
    return None, None


def _coerce_trajectory(raw: dict, source_path: str | None, source_config: str | dict | None, config_path: str | None) -> SWEAgentTrajectory:
    missing = [key for key in ("environment", "trajectory", "history", "info") if key not in raw]
    if missing:
        raise ValueError(f"SWE-agent trajectory missing required top-level keys: {', '.join(missing)}")

    trajectory = raw.get("trajectory")
    history = raw.get("history")
    info = raw.get("info")
    if not isinstance(trajectory, list):
        raise ValueError("SWE-agent trajectory field must be a list")
    if not isinstance(history, list):
        raise ValueError("SWE-agent history field must be a list")
    if not isinstance(info, dict):
        raise ValueError("SWE-agent info field must be an object")

    return SWEAgentTrajectory(
        raw=raw,
        trajectory=[item for item in trajectory if isinstance(item, dict)],
        history=[item for item in history if isinstance(item, dict)],
        info=info,
        environment=raw.get("environment"),
        source_path=source_path,
        source_config=source_config,
        source_config_path=config_path,
    )


def load_trajectory(path: str | Path, *, include_sibling_config: bool = True) -> SWEAgentTrajectory:
    trajectory_path = Path(path)
    raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SWE-agent .traj file must contain a JSON object")
    source_config: str | None = None
    config_path: str | None = None
    if include_sibling_config:
        source_config, config_path = _load_sibling_config(trajectory_path)
    return _coerce_trajectory(raw, str(trajectory_path), source_config, config_path)


def load_trajectory_json(path: str | Path, *, include_sibling_config: bool = True) -> SWEAgentTrajectory:
    return load_trajectory(path, include_sibling_config=include_sibling_config)


def _stable_episode_id(run_id: str, task_id: str, mode: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"swe-agent:{run_id}:{task_id}:{mode}"))


def _stable_missing_run_id(raw: dict) -> str:
    trajectory = raw.get("trajectory") if isinstance(raw.get("trajectory"), list) else []
    history = raw.get("history") if isinstance(raw.get("history"), list) else []
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    first_step = trajectory[0] if trajectory and isinstance(trajectory[0], dict) else {}
    last_step = trajectory[-1] if trajectory and isinstance(trajectory[-1], dict) else {}
    problem_statement = raw.get("problem_statement") if isinstance(raw.get("problem_statement"), dict) else {}
    fingerprint = {
        "environment": raw.get("environment"),
        "trajectory_len": len(trajectory),
        "history_len": len(history),
        "info_keys": sorted(str(key) for key in info.keys()),
        "exit_status": info.get("exit_status"),
        "first_action": first_step.get("action"),
        "last_action": last_step.get("action"),
        "problem_id": problem_statement.get("id"),
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"missing-run-{digest[:16]}"


def _infer_task_id(entry: SWEAgentTrajectory, explicit: str | None) -> str:
    if explicit:
        return explicit
    for value in (
        entry.info.get("instance_id"),
        entry.info.get("task_id"),
        entry.info.get("problem_id"),
        entry.raw.get("instance_id"),
        entry.raw.get("task_id"),
    ):
        if isinstance(value, str) and value:
            return value
    problem_statement = entry.raw.get("problem_statement")
    if isinstance(problem_statement, dict):
        value = problem_statement.get("id")
        if isinstance(value, str) and value:
            return value
    if entry.source_path:
        return Path(entry.source_path).stem
    return _stable_missing_run_id(entry.raw)


def _first_user_message(history: list[dict]) -> str | None:
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "human"} and isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _infer_task_description(entry: SWEAgentTrajectory, explicit: str | None, task_id: str) -> str:
    if explicit is not None:
        return explicit
    for value in (
        entry.info.get("task_description"),
        entry.info.get("problem_statement"),
        entry.raw.get("task_description"),
        entry.raw.get("problem_statement"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    problem_statement = entry.raw.get("problem_statement")
    if isinstance(problem_statement, dict):
        for key in ("text", "description", "problem_statement"):
            value = problem_statement.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return _first_user_message(entry.history) or task_id


def _infer_run_id(entry: SWEAgentTrajectory, task_id: str) -> str:
    for value in (
        entry.info.get("run_id"),
        entry.raw.get("run_id"),
        entry.info.get("trajectory_id"),
    ):
        if isinstance(value, str) and value:
            return f"swe-agent:{value}"
    if entry.source_path:
        path = Path(entry.source_path)
        return f"swe-agent:{path.parent.name}:{path.stem}"
    return f"swe-agent:{_stable_missing_run_id(entry.raw)}:{task_id}"


def _timestamps(entry: SWEAgentTrajectory, fallback: datetime = DEFAULT_TIMESTAMP) -> tuple[datetime, datetime]:
    start = None
    end = None
    for container in (entry.raw, entry.info):
        if not isinstance(container, dict):
            continue
        start = start or container.get("started_at") or container.get("start_time") or container.get("created_at")
        end = end or container.get("ended_at") or container.get("end_time") or container.get("completed_at")
    timestamp = entry.raw.get("timestamp") or entry.info.get("timestamp")
    started_at = parse_swe_agent_datetime(start or timestamp, fallback=fallback)
    ended_at = parse_swe_agent_datetime(end or timestamp, fallback=started_at)
    if ended_at == started_at:
        total = 0.0
        seen = False
        for step in entry.trajectory:
            value = step.get("execution_time")
            if isinstance(value, (int, float)):
                total += float(value)
                seen = True
        if seen:
            ended_at = started_at + timedelta(seconds=total)
    return started_at, ended_at


def _benchmark_split(entry: SWEAgentTrajectory) -> str | None:
    for value in (
        entry.info.get("benchmark_split"),
        entry.info.get("split"),
        entry.raw.get("benchmark_split"),
        entry.raw.get("split"),
    ):
        if isinstance(value, str) and value:
            return value
    if entry.source_path:
        path_text = str(entry.source_path)
        match = re.search(r"swe-bench[^/]*__(?P<config>[^/]+)__", path_text)
        if match:
            return match.group("config")
    return None


def _task_tags(entry: SWEAgentTrajectory) -> list[str] | None:
    tags: list[str] = []
    for key in ("task_tags", "tags"):
        value = entry.info.get(key) or entry.raw.get(key)
        if isinstance(value, list):
            tags.extend(str(item) for item in value if item)
    if entry.raw.get("environment"):
        tags.append(f"environment:{entry.raw['environment']}")
    return sorted(dict.fromkeys(tags)) or None


def trajectory_to_episode(
    entry: SWEAgentTrajectory,
    *,
    task_id: str | None = None,
    task_domain: str = "swe_bench",
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
    agent_surface: str = "swe_agent",
) -> Episode:
    if mode not in {"baseline", "integrated"}:
        raise ValueError("mode must be 'baseline' or 'integrated'")

    inferred_task_id = _infer_task_id(entry, task_id)
    run_id = _infer_run_id(entry, inferred_task_id)
    started_at, ended_at = _timestamps(entry)
    mapping = map_swe_agent_trajectory(
        entry.raw,
        timestamp=started_at,
        ended_at=ended_at,
        source_path=entry.source_path,
        source_config=entry.source_config,
        source_config_path=entry.source_config_path,
    )

    return Episode(
        episode_id=_stable_episode_id(run_id, inferred_task_id, mode),
        run_id=run_id,
        thread_id=inferred_task_id,
        task_id=inferred_task_id,
        task_description=_infer_task_description(entry, task_description, inferred_task_id),
        task_domain=task_domain,
        agent_surface=agent_surface,
        mode=mode,
        started_at=started_at,
        ended_at=ended_at,
        events=mapping.events,
        outcome=mapping.outcome,
        reward=None,
        repo_path=mapping.repo_path,
        base_model_id=mapping.base_model_id,
        benchmark_split=_benchmark_split(entry),
        task_tags=_task_tags(entry),
        verification_steps=mapping.verification_steps,
        verification_score=mapping.verification_score,
        verification_failures=mapping.verification_failures,
        patch_text=mapping.patch_text,
        patch_hash=mapping.patch_hash,
        tool_trace=mapping.tool_trace,
        test_trace=mapping.test_trace,
        stdout_excerpt=mapping.stdout_excerpt,
        stderr_excerpt=mapping.stderr_excerpt,
        cost_tokens_prompt=mapping.cost_tokens_prompt,
        cost_tokens_completion=mapping.cost_tokens_completion,
        latency_ms=mapping.latency_ms,
    )


def import_trajectory(
    path: str | Path,
    *,
    task_id: str | None = None,
    task_domain: str = "swe_bench",
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
    include_sibling_config: bool = True,
) -> Episode:
    return trajectory_to_episode(
        load_trajectory(path, include_sibling_config=include_sibling_config),
        task_id=task_id,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
    )


def append_trajectory_episode(
    trajectory_path: str | Path,
    episodes_path: str | Path,
    *,
    task_id: str | None = None,
    task_domain: str = "swe_bench",
    task_description: str | None = None,
    mode: Literal["baseline", "integrated"] = "baseline",
    include_sibling_config: bool = True,
) -> Episode:
    episode = import_trajectory(
        trajectory_path,
        task_id=task_id,
        task_domain=task_domain,
        task_description=task_description,
        mode=mode,
        include_sibling_config=include_sibling_config,
    )
    EpisodeRecorder(episodes_path).append(episode)
    return episode


def append_trajectory_episodes(
    trajectory_paths: list[str | Path],
    episodes_path: str | Path,
    *,
    task_domain: str = "swe_bench",
    mode: Literal["baseline", "integrated"] = "baseline",
    include_sibling_config: bool = True,
) -> list[Episode]:
    recorder = EpisodeRecorder(episodes_path)
    episodes: list[Episode] = []
    for path in trajectory_paths:
        episode = import_trajectory(
            path,
            task_domain=task_domain,
            mode=mode,
            include_sibling_config=include_sibling_config,
        )
        recorder.append(episode)
        episodes.append(episode)
    return episodes
