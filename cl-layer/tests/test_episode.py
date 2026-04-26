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


# ---------------------------------------------------------------------------
# Backward compatibility: old episodes without new fields must still load
# ---------------------------------------------------------------------------

def test_old_episode_without_new_fields_deserializes():
    """An episode serialized before the schema extension must round-trip correctly."""
    old_dict = {
        "episode_id": "ep-old",
        "run_id": "run-001",
        "thread_id": "thread-abc",
        "task_id": "task-001",
        "task_description": "Add health endpoint",
        "task_domain": "fastapi",
        "agent_surface": "codex",
        "mode": "baseline",
        "started_at": "2026-04-20T10:00:00+00:00",
        "ended_at": "2026-04-20T10:05:00+00:00",
        "events": [
            {
                "kind": "command_execution",
                "timestamp": "2026-04-20T10:00:00+00:00",
                "payload": {"command": "pytest", "exit_code": 0},
            }
        ],
        "outcome": {
            "status": "completed",
            "tests_passed": True,
            "verification_summary": "All tests passed.",
            "escalation_reason": None,
            "files_touched": ["src/app.py"],
            "final_response": "Done.",
        },
        "reward": None,
    }
    ep = episode_from_dict(old_dict)
    assert ep.episode_id == "ep-old"
    assert ep.repo_id is None
    assert ep.patch_text is None
    assert ep.verification_score is None
    assert ep.cost_tokens_prompt is None
    assert ep.candidate_rank is None
    assert ep.population_id is None


def test_old_episode_without_event_kind_still_works():
    """Events with an old kind (e.g. command_execution) should still be deserializable."""
    old_dict = {
        "episode_id": "ep-old-event",
        "run_id": "run-001",
        "thread_id": None,
        "task_id": "task-002",
        "task_description": "x",
        "task_domain": "fastapi",
        "agent_surface": "codex",
        "mode": "integrated",
        "started_at": "2026-04-20T10:00:00+00:00",
        "ended_at": "2026-04-20T10:05:00+00:00",
        "events": [
            {
                "kind": "command_execution",
                "timestamp": "2026-04-20T10:00:00+00:00",
                "payload": {},
            }
        ],
        "outcome": {
            "status": "failed",
            "tests_passed": False,
            "verification_summary": None,
            "escalation_reason": "timeout",
            "files_touched": [],
            "final_response": None,
        },
        "reward": None,
    }
    ep = episode_from_dict(old_dict)
    assert ep.outcome.status == "failed"
    assert len(ep.events) == 1


def test_backward_compat_roundtrip_through_recorder():
    """Write old-format JSON to disk, read back, verify new fields are None."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episodes.jsonl"
        old_dict = {
            "episode_id": "ep-backward",
            "run_id": "run-001",
            "thread_id": None,
            "task_id": "task-x",
            "task_description": "d",
            "task_domain": "x",
            "agent_surface": "codex",
            "mode": "baseline",
            "started_at": "2026-04-20T10:00:00+00:00",
            "ended_at": "2026-04-20T10:05:00+00:00",
            "events": [],
            "outcome": {
                "status": "completed",
                "tests_passed": True,
                "verification_summary": None,
                "escalation_reason": None,
                "files_touched": [],
                "final_response": None,
            },
            "reward": None,
        }
        with open(path, "w") as f:
            import json
            json.dump(old_dict, f)
            f.write("\n")
        loaded = EpisodeRecorder(path).load_all()
        assert len(loaded) == 1
        ep = loaded[0]
        assert ep.patch_text is None
        assert ep.verification_steps is None


# ---------------------------------------------------------------------------
# New fields roundtrip
# ---------------------------------------------------------------------------

def test_new_fields_roundtrip():
    ep = _make_episode(
        repo_id="my/repo",
        repo_path="/tmp/my-repo",
        git_commit="abc123",
        base_model_id="qwen-3b",
        student_model_id="qwen-3b-gen-1",
        parent_episode_id="ep-parent",
        benchmark_split="train",
        task_tags=["fastapi", "endpoint"],
        verification_steps=[
            {"name": "pytest", "exit_code": 0, "success": True},
            {"name": "ruff", "exit_code": 0, "success": True},
        ],
        verification_score=0.95,
        verification_failures=[],
        patch_text="--- a/src/app.py\n+++ b/src/app.py\n@@ -1,0 +1,1 @@\n+app = FastAPI()",
        patch_hash="sha256:deadbeef",
        tool_trace=[{"tool": "read_file", "args": {"path": "src/app.py"}}],
        test_trace=[{"test": "test_health", "status": "passed"}],
        stdout_excerpt="1 passed in 0.01s",
        stderr_excerpt="",
        cost_tokens_prompt=512,
        cost_tokens_completion=128,
        latency_ms=3200.5,
        candidate_rank=1,
        population_id="pop-1",
        generation_id="gen-3",
        accepted_for_training=True,
        accepted_reason="full verifier pass",
    )
    d = episode_to_dict(ep)
    ep2 = episode_from_dict(d)
    assert ep2.repo_id == "my/repo"
    assert ep2.repo_path == "/tmp/my-repo"
    assert ep2.git_commit == "abc123"
    assert ep2.base_model_id == "qwen-3b"
    assert ep2.student_model_id == "qwen-3b-gen-1"
    assert ep2.parent_episode_id == "ep-parent"
    assert ep2.benchmark_split == "train"
    assert ep2.task_tags == ["fastapi", "endpoint"]
    assert ep2.verification_score == 0.95
    assert len(ep2.verification_steps) == 2
    assert ep2.patch_text == d["patch_text"]
    assert ep2.patch_hash == "sha256:deadbeef"
    assert ep2.cost_tokens_prompt == 512
    assert ep2.cost_tokens_completion == 128
    assert ep2.latency_ms == pytest.approx(3200.5)
    assert ep2.candidate_rank == 1
    assert ep2.population_id == "pop-1"
    assert ep2.generation_id == "gen-3"
    assert ep2.accepted_for_training is True
    assert ep2.accepted_reason == "full verifier pass"


def test_new_fields_all_none_by_default():
    ep = _make_episode()
    d = episode_to_dict(ep)
    ep2 = episode_from_dict(d)
    assert ep2.repo_id is None
    assert ep2.repo_path is None
    assert ep2.git_commit is None
    assert ep2.base_model_id is None
    assert ep2.student_model_id is None
    assert ep2.parent_episode_id is None
    assert ep2.benchmark_split is None
    assert ep2.task_tags is None
    assert ep2.verification_steps is None
    assert ep2.verification_score is None
    assert ep2.verification_failures is None
    assert ep2.patch_text is None
    assert ep2.patch_hash is None
    assert ep2.tool_trace is None
    assert ep2.test_trace is None
    assert ep2.stdout_excerpt is None
    assert ep2.stderr_excerpt is None
    assert ep2.cost_tokens_prompt is None
    assert ep2.cost_tokens_completion is None
    assert ep2.latency_ms is None
    assert ep2.candidate_rank is None
    assert ep2.population_id is None
    assert ep2.generation_id is None
    assert ep2.accepted_for_training is None
    assert ep2.accepted_reason is None
