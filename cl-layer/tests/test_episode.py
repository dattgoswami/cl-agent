"""Tests for episode schema serialization, recorder, and resilience."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.episode.schema import (
    Episode,
    EpisodeEvent,
    EpisodeOutcome,
    episode_from_dict,
    episode_to_dict,
    new_episode_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event(kind: str = "command_execution") -> EpisodeEvent:
    return EpisodeEvent(
        kind=kind,
        timestamp=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        payload={"command": "pytest", "exit_code": 0},
    )


def _make_outcome(status: str = "completed") -> EpisodeOutcome:
    return EpisodeOutcome(
        status=status,
        tests_passed=True,
        verification_summary="All tests passed.",
        escalation_reason=None,
        files_touched=["src/app.py"],
        final_response="Done.",
    )


def _make_episode(**kwargs) -> Episode:
    defaults = dict(
        episode_id=new_episode_id(),
        run_id="run-001",
        thread_id="thread-abc",
        task_id="task-001",
        task_description="Add a FastAPI health endpoint",
        task_domain="fastapi",
        agent_surface="codex",
        mode="baseline",
        started_at=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 4, 20, 10, 5, 0, tzinfo=timezone.utc),
        events=[_make_event()],
        outcome=_make_outcome(),
        reward=None,
    )
    defaults.update(kwargs)
    return Episode(**defaults)


# ---------------------------------------------------------------------------
# Roundtrip serialization
# ---------------------------------------------------------------------------

def test_roundtrip_basic():
    ep = _make_episode()
    d = episode_to_dict(ep)
    ep2 = episode_from_dict(d)

    assert ep2.episode_id == ep.episode_id
    assert ep2.run_id == ep.run_id
    assert ep2.thread_id == ep.thread_id
    assert ep2.task_id == ep.task_id
    assert ep2.task_domain == ep.task_domain
    assert ep2.mode == ep.mode
    assert ep2.started_at == ep.started_at
    assert ep2.ended_at == ep.ended_at
    assert ep2.reward is None
    assert ep2.outcome.status == "completed"
    assert ep2.outcome.tests_passed is True
    assert ep2.outcome.files_touched == ["src/app.py"]
    assert len(ep2.events) == 1
    assert ep2.events[0].kind == "command_execution"
    assert ep2.events[0].payload["exit_code"] == 0


def test_roundtrip_with_reward():
    ep = _make_episode(reward=0.85)
    d = episode_to_dict(ep)
    ep2 = episode_from_dict(d)
    assert ep2.reward == pytest.approx(0.85)


def test_roundtrip_null_fields():
    ep = _make_episode(thread_id=None)
    ep.outcome.tests_passed = None
    ep.outcome.verification_summary = None
    ep.outcome.escalation_reason = None
    ep.outcome.final_response = None
    d = episode_to_dict(ep)
    ep2 = episode_from_dict(d)
    assert ep2.thread_id is None
    assert ep2.outcome.tests_passed is None
    assert ep2.outcome.final_response is None


def test_roundtrip_multiple_events():
    events = [
        _make_event("command_execution"),
        _make_event("file_change"),
        _make_event("agent_message"),
    ]
    ep = _make_episode(events=events)
    d = episode_to_dict(ep)
    ep2 = episode_from_dict(d)
    assert len(ep2.events) == 3
    assert ep2.events[1].kind == "file_change"


def test_datetime_serialized_as_isoformat():
    ep = _make_episode()
    d = episode_to_dict(ep)
    assert isinstance(d["started_at"], str)
    assert "2026-04-20" in d["started_at"]
    assert isinstance(d["events"][0]["timestamp"], str)


def test_episode_to_dict_is_json_serializable():
    ep = _make_episode()
    d = episode_to_dict(ep)
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# EpisodeRecorder
# ---------------------------------------------------------------------------

def test_recorder_append_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episodes.jsonl"
        recorder = EpisodeRecorder(path)

        ep1 = _make_episode(episode_id="ep-001", task_id="t1")
        ep2 = _make_episode(episode_id="ep-002", task_id="t2")
        recorder.append(ep1)
        recorder.append(ep2)

        loaded = recorder.load_all()
        assert len(loaded) == 2
        ids = {ep.episode_id for ep in loaded}
        assert ids == {"ep-001", "ep-002"}


def test_recorder_load_missing_file_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nonexistent.jsonl"
        recorder = EpisodeRecorder(path)
        assert recorder.load_all() == []


def test_recorder_skips_malformed_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episodes.jsonl"
        recorder = EpisodeRecorder(path)

        ep = _make_episode(episode_id="ep-good")
        recorder.append(ep)

        with open(path, "a") as f:
            f.write("THIS IS NOT JSON\n")
            f.write("{}\n")  # valid JSON but missing required keys

        loaded = recorder.load_all()
        assert len(loaded) == 1
        assert loaded[0].episode_id == "ep-good"


def test_recorder_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "deep" / "episodes.jsonl"
        recorder = EpisodeRecorder(path)
        recorder.append(_make_episode())
        assert path.exists()


def test_recorder_append_is_idempotent_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episodes.jsonl"
        EpisodeRecorder(path).append(_make_episode(episode_id="ep-1"))
        EpisodeRecorder(path).append(_make_episode(episode_id="ep-2"))
        loaded = EpisodeRecorder(path).load_all()
        assert len(loaded) == 2
