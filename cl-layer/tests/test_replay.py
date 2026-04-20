"""Tests for replay buffer ordering and filtering."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import Episode, EpisodeEvent, EpisodeOutcome, new_episode_id
from cl_layer.replay.buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def _ep(
    *,
    episode_id: str | None = None,
    task_domain: str = "fastapi",
    status: str = "completed",
    mode: str = "baseline",
    ended_offset: int = 0,
) -> Episode:
    eid = episode_id or new_episode_id()
    outcome = EpisodeOutcome(
        status=status,
        tests_passed=(status == "completed"),
        verification_summary=None,
        escalation_reason=None,
        files_touched=[],
        final_response=None,
    )
    return Episode(
        episode_id=eid,
        run_id="run-x",
        thread_id=None,
        task_id=f"task-{eid}",
        task_description="some task",
        task_domain=task_domain,
        agent_surface="codex",
        mode=mode,
        started_at=_ts(ended_offset - 1),
        ended_at=_ts(ended_offset),
        events=[],
        outcome=outcome,
        reward=None,
    )


def _recorder_with(episodes: list[Episode]) -> EpisodeRecorder:
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "episodes.jsonl"
    recorder = EpisodeRecorder(path)
    for ep in episodes:
        recorder.append(ep)
    return recorder


# ---------------------------------------------------------------------------
# Ordering tests
# ---------------------------------------------------------------------------

def test_failed_episodes_appear_in_failed_recent():
    eps = [
        _ep(episode_id="ok-1", status="completed", ended_offset=1),
        _ep(episode_id="fail-1", status="failed", ended_offset=2),
        _ep(episode_id="ok-2", status="completed", ended_offset=3),
        _ep(episode_id="fail-2", status="partial", ended_offset=4),
    ]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query()
    ids = [ep.episode_id for ep in result.failed_recent]
    assert "fail-1" in ids
    assert "fail-2" in ids
    assert "ok-1" not in ids
    assert "ok-2" not in ids


def test_failed_episodes_ordered_most_recent_first():
    eps = [
        _ep(episode_id="fail-old", status="failed", ended_offset=1),
        _ep(episode_id="fail-new", status="failed", ended_offset=10),
    ]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query()
    assert result.failed_recent[0].episode_id == "fail-new"
    assert result.failed_recent[1].episode_id == "fail-old"


def test_successes_ordered_most_recent_first():
    eps = [
        _ep(episode_id="ok-old", status="completed", ended_offset=1, task_domain="fastapi"),
        _ep(episode_id="ok-new", status="completed", ended_offset=10, task_domain="fastapi"),
    ]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query(domain="fastapi")
    assert result.success_same_domain[0].episode_id == "ok-new"
    assert result.success_same_domain[1].episode_id == "ok-old"


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------

def test_domain_filter_excludes_other_domains():
    eps = [
        _ep(episode_id="fastapi-1", task_domain="fastapi", status="completed"),
        _ep(episode_id="rust-1", task_domain="rust", status="completed"),
    ]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query(domain="fastapi")
    ids = [ep.episode_id for ep in result.success_same_domain]
    assert "fastapi-1" in ids
    assert "rust-1" not in ids


def test_domain_none_returns_all_successes():
    eps = [
        _ep(episode_id="fastapi-1", task_domain="fastapi", status="completed"),
        _ep(episode_id="rust-1", task_domain="rust", status="completed"),
    ]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query(domain=None)
    ids = [ep.episode_id for ep in result.success_same_domain]
    assert "fastapi-1" in ids
    assert "rust-1" in ids


# ---------------------------------------------------------------------------
# Mode filtering
# ---------------------------------------------------------------------------

def test_mode_filter_excludes_other_modes():
    eps = [
        _ep(episode_id="base-1", mode="baseline", status="completed"),
        _ep(episode_id="integ-1", mode="integrated", status="completed"),
    ]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query(mode="baseline")
    ids = [ep.episode_id for ep in result.success_same_domain]
    assert "base-1" in ids
    assert "integ-1" not in ids


# ---------------------------------------------------------------------------
# max_failures / max_successes caps
# ---------------------------------------------------------------------------

def test_max_failures_cap():
    eps = [_ep(episode_id=f"fail-{i}", status="failed", ended_offset=i) for i in range(10)]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query(max_failures=3)
    assert len(result.failed_recent) == 3


def test_max_successes_cap():
    eps = [_ep(episode_id=f"ok-{i}", status="completed", ended_offset=i) for i in range(10)]
    buf = ReplayBuffer(_recorder_with(eps))
    result = buf.query(max_successes=4)
    assert len(result.success_same_domain) == 4


# ---------------------------------------------------------------------------
# query_by_task / query_by_domain
# ---------------------------------------------------------------------------

def test_query_by_task():
    ep1 = _ep(episode_id="ep-1")
    ep1.task_id = "target-task"
    ep2 = _ep(episode_id="ep-2")
    ep2.task_id = "other-task"
    buf = ReplayBuffer(_recorder_with([ep1, ep2]))
    found = buf.query_by_task("target-task")
    assert len(found) == 1
    assert found[0].episode_id == "ep-1"


def test_query_by_domain():
    eps = [
        _ep(episode_id="d1", task_domain="fastapi"),
        _ep(episode_id="d2", task_domain="rust"),
        _ep(episode_id="d3", task_domain="fastapi"),
    ]
    buf = ReplayBuffer(_recorder_with(eps))
    found = buf.query_by_domain("fastapi")
    ids = {ep.episode_id for ep in found}
    assert ids == {"d1", "d3"}


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

def test_empty_store_returns_empty_result():
    buf = ReplayBuffer(_recorder_with([]))
    result = buf.query()
    assert result.failed_recent == []
    assert result.success_same_domain == []
