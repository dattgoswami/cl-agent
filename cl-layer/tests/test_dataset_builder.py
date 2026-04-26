"""Tests for dataset builder package."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cl_layer.dataset.example_schema import ExampleType, TrainingExample, make_example_id
from cl_layer.dataset.from_episode import episode_to_example
from cl_layer.dataset.hindsight import relabel_partial_success, relabeled_to_examples
from cl_layer.dataset.dedup import dedup_by_normalized_text, dedup_by_patch_hash, dedup_examples, normalize_patch
from cl_layer.dataset.filters import filter_examples, reject_empty_target, reject_giant_diff, reject_hidden_state, reject_no_verifier
from cl_layer.dataset.splits import split_datasets
from cl_layer.dataset.render_chat import render_example_chat, render_examples_chatl, DEFAULT_CHAT_TEMPLATE
from cl_layer.episode.schema import Episode, EpisodeEvent, EpisodeOutcome, new_episode_id


# --------------- fixtures ------------

def _make_episode(
    verification_score: float | None = 0.9,
    patch_text: str | None = "diff --git a/x.py b/x.py\n+print('hello')",
    test_trace: list | None = None,
    stderr_excerpt: str | None = None,
    **kwargs,
) -> Episode:
    return Episode(
        episode_id=kwargs.pop("episode_id", new_episode_id()),
        run_id="run-001",
        thread_id="thread-1",
        task_id=kwargs.pop("task_id", "task-001"),
        task_description=kwargs.pop("task_description", "Add health endpoint"),
        task_domain=kwargs.pop("task_domain", "fastapi"),
        agent_surface="codex",
        mode="baseline",
        started_at=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 4, 20, 10, 5, 0, tzinfo=timezone.utc),
        events=[
            EpisodeEvent(
                kind="command_execution",
                timestamp=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
                payload={"command": "pytest", "exit_code": 0},
            )
        ],
        outcome=EpisodeOutcome(
            status="completed",
            tests_passed=True,
            verification_summary="All tests passed.",
            escalation_reason=None,
            files_touched=["src/app.py"],
            final_response="Done.",
        ),
        reward=None,
        verification_score=verification_score,
        patch_text=patch_text,
        test_trace=test_trace,
        stderr_excerpt=stderr_excerpt,
        **kwargs,
    )


def _make_example(
    input_text: str = "task prompt",
    target_text: str = "target text",
    **kwargs,
) -> TrainingExample:
    return TrainingExample(
        id=make_example_id(input_text, target_text),
        input_text=input_text,
        target_text=target_text,
        **kwargs,
    )


# --------------- example_schema ------------

class TestTrainingExample:
    def test_make_example_id_stable(self):
        id1 = make_example_id("a", "b")
        id2 = make_example_id("a", "b")
        assert id1 == id2

    def test_make_example_id_different(self):
        id1 = make_example_id("a", "b")
        id2 = make_example_id("c", "d")
        assert id1 != id2

    def test_roundtrip_to_dict(self):
        ex = _make_example()
        d = ex.to_dict()
        ex2 = TrainingExample.from_dict(d)
        assert ex2.id == ex.id
        assert ex2.input_text == ex.input_text
        assert ex2.target_text == ex.target_text
        assert ex2.example_type == ex.example_type
        assert ex2.source_episode_id == ex.source_episode_id

    def test_to_dict_is_json_serializable(self):
        ex = _make_example()
        d = ex.to_dict()
        json.dumps(d)


# --------------- from_episode ------------

class TestEpisodeToExample:
    def test_prefers_patch_text(self):
        ep = _make_episode()
        ex = episode_to_example(ep)
        assert ex is not None
        assert ex.target_text == ep.patch_text

    def test_returns_none_on_zero_score(self):
        ep = _make_episode(verification_score=0.0)
        assert episode_to_example(ep) is None

    def test_returns_none_on_no_patch_and_no_trace(self):
        ep = _make_episode(patch_text=None)
        assert episode_to_example(ep) is None

    def test_sets_source_episode_id(self):
        ep = _make_episode(episode_id="ep-123")
        ex = episode_to_example(ep)
        assert ex.source_episode_id == "ep-123"

    def test_falls_back_to_tool_trace(self):
        ep = _make_episode(patch_text=None, tool_trace=[{"tool": "read"}])
        ex = episode_to_example(ep)
        assert ex is not None
        assert "read" in ex.target_text


# --------------- hindsight ------------

class TestHindsight:
    def test_relabel_from_test_trace(self):
        ep = _make_episode(
            test_trace=[{"test": "test_health", "status": "passed"}],
        )
        subtasks = relabel_partial_success(ep)
        assert len(subtasks) >= 1
        assert subtasks[0].confidence >= 0.8

    def test_relabel_from_stderr_import_error(self):
        ep = _make_episode(
            stderr_excerpt="ImportError: cannot import name 'FastAPI'",
        )
        subtasks = relabel_partial_success(ep)
        assert any("import" in s.task_description.lower() for s in subtasks)

    def test_relabel_from_partial_score(self):
        ep = _make_episode(verification_score=0.5)
        subtasks = relabel_partial_success(ep)
        assert any(s.confidence == 0.5 for s in subtasks)

    def test_no_patch_returns_empty(self):
        ep = _make_episode(patch_text=None)
        assert relabel_partial_success(ep) == []

    def test_relabeled_to_examples(self):
        ep = _make_episode(test_trace=[{"test": "test_a", "status": "passed"}])
        examples = relabeled_to_examples(ep)
        assert len(examples) >= 1
        assert examples[0].example_type == ExampleType.subtask


# --------------- dedup ------------

class TestDedup:
    def test_normalize_patch_collapses_whitespace(self):
        assert normalize_patch("a  b\nc") == "a b c"

    def test_normalize_patch_removes_diff_headers(self):
        text = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-hello\n+world"
        normalized = normalize_patch(text)
        assert "+++" not in normalized
        assert "---" not in normalized

    def test_dedup_by_patch_hash_duplicates(self):
        ex1 = _make_example(target_text="patch1")
        ex1.metadata["patch_hash"] = "hash-a"
        ex2 = _make_example(target_text="different")
        ex2.metadata["patch_hash"] = "hash-a"
        deduped = dedup_by_patch_hash([ex1, ex2])
        assert len(deduped) == 1

    def test_dedup_by_normalized_text_duplicates(self):
        ex1 = _make_example(target_text="a  b\nc")
        ex2 = _make_example(target_text="a b c")
        deduped = dedup_by_normalized_text([ex1, ex2])
        assert len(deduped) == 1

    def test_dedup_examples_returns_counts(self):
        ex1 = _make_example(target_text="patch1")
        ex1.metadata["patch_hash"] = "hash-a"
        ex2 = _make_example(target_text="different")
        ex2.metadata["patch_hash"] = "hash-a"
        deduped, counts = dedup_examples([ex1, ex2])
        assert counts["input"] == 2
        assert counts["after_patch_hash"] == 1


# --------------- filters ------------

class TestFilters:
    def test_reject_no_verifier(self):
        ex = _make_example(metadata={"verification_score": None})
        assert reject_no_verifier(ex) is True

    def test_reject_empty_target(self):
        ex = _make_example(target_text="")
        assert reject_empty_target(ex) is True

    def test_reject_giant_diff(self):
        big = "\n@@hunk".join([""] * 60)
        ex = _make_example(target_text=big)
        assert reject_giant_diff(ex) is True

    def test_reject_hidden_state(self):
        ex = _make_example(target_text="import os; os.urandom(16)")
        assert reject_hidden_state(ex) is True

    def test_filter_examples(self):
        examples = [
            _make_example(target_text="good", metadata={"verification_score": 0.9}),
            _make_example(target_text="", metadata={"verification_score": 0.9}),
        ]
        filtered, counts = filter_examples(examples)
        assert counts["input"] == 2
        assert counts["output"] == 1
        assert counts["rejected_empty_target"] == 1


# --------------- splits ------------

class TestSplits:
    def test_split_proportions(self):
        examples = [_make_example(target_text=f"ex{i}") for i in range(100)]
        train, valid, test = split_datasets(examples)
        assert len(train) == 70
        assert len(valid) == 15
        assert len(test) == 15

    def test_split_empty(self):
        train, valid, test = split_datasets([])
        assert train == []
        assert valid == []
        assert test == []

    def test_split_deterministic(self):
        examples = [_make_example(target_text=f"ex{i}") for i in range(20)]
        t1, v1, te1 = split_datasets(examples)
        t2, v2, te2 = split_datasets(examples)
        assert [e.id for e in t1] == [e.id for e in t2]

    def test_split_full_list(self):
        examples = [_make_example(target_text=f"ex{i}") for i in range(10)]
        train, valid, test = split_datasets(examples, train_ratio=0.5, valid_ratio=0.3, test_ratio=0.2)
        assert len(train) == 5
        assert len(valid) == 3
        assert len(test) == 2


# --------------- render_chat ------------

class TestRenderChat:
    def test_render_example_chat(self):
        ex = _make_example(input_text="do this", target_text="result")
        rendered = render_example_chat(ex)
        assert "do this" in rendered
        assert "result" in rendered

    def test_render_examples_chatl(self):
        examples = [
            _make_example(input_text="a", target_text="b"),
            _make_example(input_text="c", target_text="d"),
        ]
        rendered = render_examples_chatl(examples)
        assert len(rendered) == 2

    def test_default_template_has_system(self):
        assert "system" in DEFAULT_CHAT_TEMPLATE.system_prompt.lower() or "system" in DEFAULT_CHAT_TEMPLATE.system_prompt
