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
from cl_layer.dataset.render_chat import (
    DEFAULT_CHAT_TEMPLATE,
    ChatTemplate,
    example_to_messages,
    render_example_chat,
    render_example_jsonl,
    render_examples_chatl,
    render_messages_chatml,
)
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

    def test_returns_none_on_missing_patch(self):
        ep = _make_episode(patch_text=None)
        assert episode_to_example(ep) is None

    def test_returns_none_on_empty_patch(self):
        ep = _make_episode(patch_text="")
        assert episode_to_example(ep) is None

    def test_returns_none_on_none_verification_score(self):
        # Strict: missing verifier outcome means the episode is not eligible.
        ep = _make_episode(verification_score=None)
        assert episode_to_example(ep) is None

    def test_returns_none_on_negative_verification_score(self):
        ep = _make_episode(verification_score=-0.1)
        assert episode_to_example(ep) is None

    def test_does_not_fall_back_to_tool_trace(self):
        # tool_trace is no longer accepted as a target — patch_text is required.
        ep = _make_episode(patch_text=None, tool_trace=[{"tool": "read"}])
        assert episode_to_example(ep) is None

    def test_sets_source_episode_id(self):
        ep = _make_episode(episode_id="ep-123")
        ex = episode_to_example(ep)
        assert ex.source_episode_id == "ep-123"

    def test_propagates_patch_hash_to_metadata(self):
        ep = _make_episode(patch_hash="sha256:deadbeef")
        ex = episode_to_example(ep)
        assert ex is not None
        assert ex.metadata["patch_hash"] == "sha256:deadbeef"

    def test_omits_patch_hash_when_absent(self):
        ep = _make_episode(patch_hash=None)
        ex = episode_to_example(ep)
        assert ex is not None
        assert "patch_hash" not in ex.metadata

    def test_preserves_task_id_and_domain(self):
        ep = _make_episode(task_id="task-xyz", task_domain="auth")
        ex = episode_to_example(ep)
        assert ex is not None
        assert ex.metadata["task_id"] == "task-xyz"
        assert ex.metadata["task_domain"] == "auth"

    def test_preserves_generation_id_and_score(self):
        ep = _make_episode(generation_id="gen-7", verification_score=0.91)
        ex = episode_to_example(ep)
        assert ex is not None
        assert ex.metadata["generation_id"] == "gen-7"
        assert ex.metadata["verification_score"] == pytest.approx(0.91)


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

class TestChatTemplate:
    def test_default_template_role_markers(self):
        assert DEFAULT_CHAT_TEMPLATE.role_start == "<|im_start|>"
        assert DEFAULT_CHAT_TEMPLATE.role_end == "<|im_end|>"

    def test_default_template_has_nonempty_system_prompt(self):
        assert isinstance(DEFAULT_CHAT_TEMPLATE.system_prompt, str)
        assert DEFAULT_CHAT_TEMPLATE.system_prompt.strip() != ""

    def test_eos_token_alias(self):
        assert DEFAULT_CHAT_TEMPLATE.eos_token == DEFAULT_CHAT_TEMPLATE.role_end

    def test_stop_tokens_includes_both_boundaries(self):
        stops = DEFAULT_CHAT_TEMPLATE.stop_tokens
        assert "<|im_end|>" in stops
        assert "<|im_start|>" in stops


class TestRenderChat:
    def test_example_to_messages_has_three_roles(self):
        ex = _make_example(input_text="u", target_text="a")
        msgs = example_to_messages(ex)
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[1]["content"] == "u"
        assert msgs[2]["content"] == "a"

    def test_render_example_chat_golden(self):
        """Render against an exact expected ChatML string — catches any drift
        in the template that would desync trainer and Ollama runtime."""
        ex = _make_example(input_text="do this", target_text="result")
        tmpl = ChatTemplate(system_prompt="SYS")
        rendered = render_example_chat(ex, tmpl)
        expected = (
            "<|im_start|>system\nSYS<|im_end|>\n"
            "<|im_start|>user\ndo this<|im_end|>\n"
            "<|im_start|>assistant\nresult<|im_end|>"
        )
        assert rendered == expected

    def test_render_example_chat_no_double_eos(self):
        ex = _make_example(input_text="do this", target_text="result")
        rendered = render_example_chat(ex)
        assert "<|im_end|><|im_end|>" not in rendered
        # exactly three turns => exactly three role_end tokens
        assert rendered.count("<|im_end|>") == 3
        assert rendered.count("<|im_start|>") == 3

    def test_render_messages_chatml_arbitrary_turns(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        out = render_messages_chatml(msgs)
        assert out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhey<|im_end|>"
        assert "<|im_end|><|im_end|>" not in out

    def test_render_example_jsonl_parses(self):
        ex = _make_example(input_text="u", target_text="a")
        line = render_example_jsonl(ex)
        record = json.loads(line)
        assert "messages" in record
        roles = [m["role"] for m in record["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert record["messages"][1]["content"] == "u"
        assert record["messages"][2]["content"] == "a"

    def test_render_examples_chatl_returns_jsonl(self):
        examples = [
            _make_example(input_text="a", target_text="b"),
            _make_example(input_text="c", target_text="d"),
        ]
        rendered = render_examples_chatl(examples)
        assert len(rendered) == 2
        # Each line must be a valid JSON object with the expected shape.
        for line, ex in zip(rendered, examples):
            assert "\n" not in line  # one JSON object per line, no embedded newlines
            obj = json.loads(line)
            assert list(obj.keys()) == ["messages"]
            roles = [m["role"] for m in obj["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert obj["messages"][1]["content"] == ex.input_text
            assert obj["messages"][2]["content"] == ex.target_text

    def test_render_examples_chatl_writable_to_file(self, tmp_path):
        examples = [_make_example(input_text=f"u{i}", target_text=f"a{i}") for i in range(3)]
        lines = render_examples_chatl(examples)
        path = tmp_path / "train.jsonl"
        path.write_text("\n".join(lines) + "\n")
        # Read back and re-parse, verify all 3 records survive.
        loaded = [json.loads(l) for l in path.read_text().splitlines() if l]
        assert len(loaded) == 3
        for i, rec in enumerate(loaded):
            assert rec["messages"][1]["content"] == f"u{i}"
            assert rec["messages"][2]["content"] == f"a{i}"

    def test_custom_system_prompt_propagates_to_messages(self):
        ex = _make_example()
        tmpl = ChatTemplate(system_prompt="custom")
        msgs = example_to_messages(ex, tmpl)
        assert msgs[0]["content"] == "custom"


# --------------- splits: leakage prevention ------------

class TestSplitsTaskLeakage:
    def _ex_with_task(self, task_id: str, target: str) -> TrainingExample:
        return TrainingExample(
            id=make_example_id(task_id, target),
            input_text=f"task {task_id}",
            target_text=target,
            metadata={"task_id": task_id},
        )

    def test_same_task_id_never_in_two_splits(self):
        # 50 tasks, each with 5 different patches solving it. If splits
        # leaked, the same task_id would appear in train AND test.
        examples: list[TrainingExample] = []
        for t in range(50):
            for k in range(5):
                examples.append(self._ex_with_task(f"task-{t}", f"patch-{t}-{k}"))
        train, valid, test = split_datasets(examples)
        train_tasks = {ex.metadata["task_id"] for ex in train}
        valid_tasks = {ex.metadata["task_id"] for ex in valid}
        test_tasks = {ex.metadata["task_id"] for ex in test}
        assert train_tasks.isdisjoint(valid_tasks)
        assert train_tasks.isdisjoint(test_tasks)
        assert valid_tasks.isdisjoint(test_tasks)
        assert train_tasks | valid_tasks | test_tasks == {f"task-{t}" for t in range(50)}

    def test_split_stable_under_reordering(self):
        examples = [self._ex_with_task(f"task-{t}", f"p-{t}") for t in range(40)]
        a = split_datasets(examples)
        b = split_datasets(list(reversed(examples)))
        # Same task_id assignment regardless of input order.
        assert {e.metadata["task_id"] for e in a[0]} == {e.metadata["task_id"] for e in b[0]}
        assert {e.metadata["task_id"] for e in a[1]} == {e.metadata["task_id"] for e in b[1]}
        assert {e.metadata["task_id"] for e in a[2]} == {e.metadata["task_id"] for e in b[2]}


# --------------- builder ------------

from cl_layer.dataset.builder import DatasetManifest, build_dataset
from cl_layer.dataset.splits import SplitConfig


def _verified_episode(
    *,
    episode_id: str,
    task_id: str,
    patch_text: str,
    patch_hash: str | None = None,
    verification_score: float = 0.9,
    task_description: str = "do the thing",
    task_domain: str = "fastapi",
    generation_id: str | None = "gen-0001",
) -> Episode:
    return Episode(
        episode_id=episode_id,
        run_id="run-001",
        thread_id=None,
        task_id=task_id,
        task_description=task_description,
        task_domain=task_domain,
        agent_surface="codex",
        mode="baseline",
        started_at=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 4, 20, 10, 5, 0, tzinfo=timezone.utc),
        events=[],
        outcome=EpisodeOutcome(
            status="completed",
            tests_passed=True,
            verification_summary=None,
            escalation_reason=None,
            files_touched=[],
            final_response=None,
        ),
        verification_score=verification_score,
        patch_text=patch_text,
        patch_hash=patch_hash,
        generation_id=generation_id,
    )


class TestBuildDataset:
    def test_writes_all_four_files(self, tmp_path):
        episodes = [
            _verified_episode(episode_id=f"ep-{i}", task_id=f"task-{i}", patch_text=f"diff {i}")
            for i in range(20)
        ]
        manifest = build_dataset(episodes, tmp_path, "gen-0001")
        out = tmp_path / "gen-0001"
        assert (out / "train.jsonl").exists()
        assert (out / "valid.jsonl").exists()
        assert (out / "test.jsonl").exists()
        assert (out / "manifest.json").exists()
        assert isinstance(manifest, DatasetManifest)

    def test_jsonl_files_are_parseable(self, tmp_path):
        episodes = [
            _verified_episode(episode_id=f"ep-{i}", task_id=f"task-{i}", patch_text=f"diff {i}")
            for i in range(20)
        ]
        build_dataset(episodes, tmp_path, "gen-0001")
        out = tmp_path / "gen-0001"
        for name in ("train", "valid", "test"):
            text = (out / f"{name}.jsonl").read_text()
            for line in text.splitlines():
                rec = json.loads(line)
                assert "messages" in rec
                roles = [m["role"] for m in rec["messages"]]
                assert roles == ["system", "user", "assistant"]

    def test_manifest_shape(self, tmp_path):
        episodes = [
            _verified_episode(episode_id=f"ep-{i}", task_id=f"task-{i}", patch_text=f"diff {i}")
            for i in range(20)
        ]
        build_dataset(episodes, tmp_path, "gen-0001")
        manifest = json.loads((tmp_path / "gen-0001" / "manifest.json").read_text())
        assert manifest["gen_id"] == "gen-0001"
        assert "created_at" in manifest
        assert "template" in manifest
        assert manifest["template"]["role_start"] == "<|im_start|>"
        assert manifest["template"]["role_end"] == "<|im_end|>"
        assert "source_episode_ids" in manifest
        assert "counts" in manifest
        assert "splits" in manifest
        for name in ("train", "valid", "test"):
            assert name in manifest["splits"]
            assert "size" in manifest["splits"][name]
            assert "path" in manifest["splits"][name]
            assert "example_ids" in manifest["splits"][name]

    def test_source_episode_ids_recorded(self, tmp_path):
        episodes = [
            _verified_episode(episode_id=f"ep-{i}", task_id=f"task-{i}", patch_text=f"diff {i}")
            for i in range(5)
        ]
        manifest = build_dataset(episodes, tmp_path, "gen-0001")
        assert set(manifest.source_episode_ids) == {f"ep-{i}" for i in range(5)}

    def test_split_sizes_sum_to_eligible_inputs(self, tmp_path):
        episodes = [
            _verified_episode(episode_id=f"ep-{i}", task_id=f"task-{i}", patch_text=f"diff {i}")
            for i in range(30)
        ]
        manifest = build_dataset(episodes, tmp_path, "gen-0001")
        total = (
            manifest.splits["train"]["size"]
            + manifest.splits["valid"]["size"]
            + manifest.splits["test"]["size"]
        )
        assert total == manifest.counts["examples_total_in_splits"]
        assert total == manifest.counts["examples_after_dedup"]

    def test_rejects_episode_with_none_verification_score(self, tmp_path):
        episodes = [
            _verified_episode(episode_id="ep-good", task_id="task-1", patch_text="diff a"),
            _verified_episode(
                episode_id="ep-bad",
                task_id="task-2",
                patch_text="diff b",
                verification_score=None,  # type: ignore[arg-type]
            ),
        ]
        manifest = build_dataset(episodes, tmp_path, "gen-0001")
        assert "ep-good" in manifest.source_episode_ids
        assert "ep-bad" not in manifest.source_episode_ids
        assert manifest.counts["episodes_rejected"] == 1

    def test_rejects_episode_with_zero_score(self, tmp_path):
        episodes = [
            _verified_episode(
                episode_id="ep-zero", task_id="t", patch_text="d", verification_score=0.0
            ),
        ]
        manifest = build_dataset(episodes, tmp_path, "gen-0001")
        assert manifest.counts["episodes_rejected"] == 1
        assert manifest.source_episode_ids == []

    def test_rejects_episode_with_no_patch(self, tmp_path):
        ep = _verified_episode(episode_id="ep-no-patch", task_id="t", patch_text="")
        manifest = build_dataset([ep], tmp_path, "gen-0001")
        assert manifest.counts["episodes_rejected"] == 1

    def test_dedup_by_patch_hash_collapses_duplicates(self, tmp_path):
        # Two different episodes with identical patch_hash should collapse to one example.
        episodes = [
            _verified_episode(
                episode_id="ep-a",
                task_id="task-a",
                patch_text="diff alpha v1",
                patch_hash="sha256:same",
            ),
            _verified_episode(
                episode_id="ep-b",
                task_id="task-b",
                patch_text="diff alpha v2",  # different text, same hash
                patch_hash="sha256:same",
            ),
            _verified_episode(
                episode_id="ep-c",
                task_id="task-c",
                patch_text="diff beta",
                patch_hash="sha256:other",
            ),
        ]
        manifest = build_dataset(episodes, tmp_path, "gen-0001")
        # 3 episodes converted; patch-hash dedup drops one collision; 2 survive.
        assert manifest.counts["examples_converted"] == 3
        assert manifest.counts["dedup_after_patch_hash"] == 2
        assert manifest.counts["examples_total_in_splits"] == 2

    def test_no_task_leakage_across_splits(self, tmp_path):
        # Multiple distinct patches per task — splits must keep them together.
        episodes: list[Episode] = []
        for t in range(30):
            for k in range(3):
                episodes.append(
                    _verified_episode(
                        episode_id=f"ep-{t}-{k}",
                        task_id=f"task-{t}",
                        patch_text=f"diff t{t} k{k}",
                    )
                )
        build_dataset(episodes, tmp_path, "gen-0001")
        out = tmp_path / "gen-0001"

        def tasks_in(path: Path) -> set[str]:
            tasks: set[str] = set()
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                user = rec["messages"][1]["content"]
                # input_text starts with task_description "do the thing"; we need
                # task identity from somewhere — pull it from the input_text via
                # the description prefix. We'll instead re-derive from manifest.
                tasks.add(user)
            return tasks

        manifest = json.loads((out / "manifest.json").read_text())
        # Cross-check by example_ids: the same example never appears twice
        all_ids: list[str] = []
        for name in ("train", "valid", "test"):
            all_ids.extend(manifest["splits"][name]["example_ids"])
        assert len(all_ids) == len(set(all_ids))

        # And: tasks (derived from example metadata) are disjoint per split.
        # We rebuild this using the deterministic split helper directly, since
        # JSONL records don't carry metadata.
        from cl_layer.dataset.from_episode import episode_to_example
        examples = [episode_to_example(ep) for ep in episodes]
        examples = [e for e in examples if e is not None]
        train, valid, test = split_datasets(examples)
        train_t = {e.metadata["task_id"] for e in train}
        valid_t = {e.metadata["task_id"] for e in valid}
        test_t = {e.metadata["task_id"] for e in test}
        assert train_t.isdisjoint(valid_t)
        assert train_t.isdisjoint(test_t)
        assert valid_t.isdisjoint(test_t)

    def test_filter_counts_recorded_in_manifest(self, tmp_path):
        # Build episodes; then directly inject one giant-diff and one
        # hidden-state example via filters by giving an episode a target that
        # filters will catch. Since filters operate on examples, the simplest
        # test is via episode patch_text content.
        big_patch = "header\n" + ("\n@@hunk" * 60)
        episodes = [
            _verified_episode(episode_id="ep-good", task_id="g", patch_text="diff a"),
            _verified_episode(episode_id="ep-big", task_id="b", patch_text=big_patch),
            _verified_episode(
                episode_id="ep-hidden",
                task_id="h",
                patch_text="something with os.urandom inside",
            ),
        ]
        manifest = build_dataset(episodes, tmp_path, "gen-0001")
        assert manifest.counts["rejected_giant_diff"] >= 1
        assert manifest.counts["rejected_hidden_state"] >= 1

    def test_repeated_runs_produce_same_split_assignment(self, tmp_path):
        episodes = [
            _verified_episode(episode_id=f"ep-{t}", task_id=f"task-{t}", patch_text=f"d-{t}")
            for t in range(40)
        ]
        m1 = build_dataset(episodes, tmp_path / "a", "gen-0001")
        m2 = build_dataset(episodes, tmp_path / "b", "gen-0001")
        for name in ("train", "valid", "test"):
            assert m1.splits[name]["example_ids"] == m2.splits[name]["example_ids"]

    def test_empty_episode_list_creates_empty_splits(self, tmp_path):
        manifest = build_dataset([], tmp_path, "gen-0001")
        out = tmp_path / "gen-0001"
        for name in ("train", "valid", "test"):
            assert (out / f"{name}.jsonl").read_text() == ""
        assert manifest.counts["examples_total_in_splits"] == 0

    def test_uses_provided_template_in_manifest(self, tmp_path):
        episodes = [
            _verified_episode(episode_id="ep-1", task_id="t-1", patch_text="d")
        ]
        tmpl = ChatTemplate(system_prompt="custom-sys-msg")
        manifest = build_dataset(episodes, tmp_path, "gen-0001", template=tmpl)
        assert manifest.template["system_prompt"] == "custom-sys-msg"
