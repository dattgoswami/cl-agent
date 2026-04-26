"""Phase-1 happy-path integration smoke test.

Wires the corrected dataset orchestrator, the MLX trainer shell with a
recording fake runner, the GGUF export step, and the Ollama Modelfile
generator into a single synthetic end-to-end run. Pure in-memory and
in-tempdir — no MLX, Torch, requests, Ollama, network, or heavyweight ML
deps are imported or required.

This test is the safety net for the three regressions called out in
``findings.md``:
  - the Modelfile triple-quote bug (Finding A)
  - the dataset orchestration gap (Finding D)
  - the MLX backend's runner no-op (Finding B)

If any of those regress, this test fails.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cl_layer.dataset import (
    ChatTemplate,
    DatasetManifest,
    SplitConfig,
    build_dataset,
)
from cl_layer.episode.schema import Episode, EpisodeOutcome
from cl_layer.serve.modelfile import generate_modelfile
from cl_layer.train.base import SmokeResult, TrainConfig
from cl_layer.train.mlx_backend import MLXTrainerBackend


# --------------- helpers ------------


def _episode(
    *,
    episode_id: str,
    task_id: str,
    patch_text: str,
    patch_hash: str,
    score: float = 0.9,
    domain: str = "fastapi",
) -> Episode:
    return Episode(
        episode_id=episode_id,
        run_id="run-001",
        thread_id=None,
        task_id=task_id,
        task_description=f"Solve {task_id}",
        task_domain=domain,
        agent_surface="codex",
        mode="baseline",
        started_at=datetime(2026, 4, 25, 10, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 4, 25, 10, 5, 0, tzinfo=timezone.utc),
        events=[],
        outcome=EpisodeOutcome(
            status="completed",
            tests_passed=True,
            verification_summary=None,
            escalation_reason=None,
            files_touched=[],
            final_response=None,
        ),
        verification_score=score,
        patch_text=patch_text,
        patch_hash=patch_hash,
        generation_id="gen-0001",
    )


def _synthetic_episodes() -> list[Episode]:
    """Six episodes across three task ids and four distinct patch hashes.

    - task-a has two episodes with the same patch_hash (dedup target).
    - task-b has two episodes with different hashes.
    - task-c has two episodes with different hashes.
    """
    return [
        _episode(
            episode_id=f"ep-a-{i}",
            task_id="task-a",
            patch_text=f"diff a v{i}",
            patch_hash="sha256:aaaa",
        )
        for i in range(2)
    ] + [
        _episode(
            episode_id="ep-b-1",
            task_id="task-b",
            patch_text="diff b1",
            patch_hash="sha256:bbb1",
        ),
        _episode(
            episode_id="ep-b-2",
            task_id="task-b",
            patch_text="diff b2",
            patch_hash="sha256:bbb2",
        ),
        _episode(
            episode_id="ep-c-1",
            task_id="task-c",
            patch_text="diff c1",
            patch_hash="sha256:ccc1",
        ),
        _episode(
            episode_id="ep-c-2",
            task_id="task-c",
            patch_text="diff c2",
            patch_hash="sha256:ccc2",
        ),
    ]


class _RecordingRunner:
    """Fake MLX runner that records every call and returns deterministic paths."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.calls: list[tuple[str, dict]] = []

    def load_model(self, model_id: str, output_dir: Path) -> dict:
        self.calls.append(
            ("load_model", {"model_id": model_id, "output_dir": str(output_dir)})
        )
        return {"model_id": model_id, "output_dir": str(output_dir)}

    def train_lora(self, handle: dict, config, output_dir: Path) -> dict:
        self.calls.append(
            (
                "train_lora",
                {
                    "model_id": config.model_id,
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "lora_rank": config.lora_rank,
                    "output_dir": str(output_dir),
                },
            )
        )
        adapter = self._root / "fake_adapter"
        return {
            "adapter_dir": str(adapter),
            "metrics": {"loss": 0.123, "step": 1000},
        }

    def fuse_model(self, handle: dict, train_result: dict, output_dir: Path) -> dict:
        self.calls.append(
            (
                "fuse_model",
                {
                    "model_id": handle.get("model_id"),
                    "adapter_dir": handle.get("adapter_dir"),
                    "output_dir": str(output_dir),
                },
            )
        )
        return {"export_dir": str(output_dir)}

    def convert_gguf(self, handle: dict, quant: str, output_dir: Path) -> Path:
        self.calls.append(
            ("convert_gguf", {"quant": quant, "output_dir": str(output_dir)})
        )
        return output_dir / f"model-{quant}.gguf"

    def smoke_test(self, model_path: Path, prompt_set: Path) -> SmokeResult:
        self.calls.append(
            ("smoke_test", {"model_path": str(model_path), "prompt_set": str(prompt_set)})
        )
        return SmokeResult(
            model_path=str(model_path),
            prompts_tested=1,
            prompts_passed=1,
            latency_ms=42.0,
            passed=True,
        )


# --------------- the integration test ------------


class TestPhase1HappyPath:
    """End-to-end: synthetic episodes -> JSONL splits + manifest -> mocked
    MLX train/fuse/gguf -> Ollama Modelfile."""

    def test_episode_to_modelfile_pipeline(self, tmp_path):
        # 1. Synthesize verified episodes.
        episodes = _synthetic_episodes()
        assert len({ep.task_id for ep in episodes}) >= 2
        assert len({ep.patch_hash for ep in episodes}) >= 2

        # 2. Run the dataset builder.
        dataset_root = tmp_path / "data" / "datasets"
        chat_template = ChatTemplate(system_prompt="You write small patches.")
        manifest = build_dataset(
            episodes,
            dataset_root,
            "gen-0001",
            template=chat_template,
            split_config=SplitConfig(
                train_ratio=0.5, valid_ratio=0.25, test_ratio=0.25
            ),
        )
        assert isinstance(manifest, DatasetManifest)
        gen_dir = dataset_root / "gen-0001"

        # 3. All four files present.
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "manifest.json"):
            assert (gen_dir / name).exists(), f"missing {name}"

        # 4. Every JSONL line parses and has the canonical shape.
        per_split_counts = {}
        for split in ("train", "valid", "test"):
            text = (gen_dir / f"{split}.jsonl").read_text()
            lines = [l for l in text.splitlines() if l]
            for line in lines:
                rec = json.loads(line)
                assert list(rec.keys()) == ["messages"]
                roles = [m["role"] for m in rec["messages"]]
                assert roles == ["system", "user", "assistant"]
                # System prompt comes from the supplied template.
                assert rec["messages"][0]["content"] == "You write small patches."
                # No double EOS even after JSON encoding.
                assert "<|im_end|><|im_end|>" not in line
            per_split_counts[split] = len(lines)

        # 5. Manifest split sizes match file contents AND the reported total.
        for split, count in per_split_counts.items():
            assert manifest.splits[split]["size"] == count
        assert (
            sum(per_split_counts.values())
            == manifest.counts["examples_total_in_splits"]
        )

        # ep-a-0 and ep-a-1 share patch_hash => one of them dedups out.
        assert manifest.counts["examples_converted"] == 6
        assert manifest.counts["dedup_after_patch_hash"] == 5

        # No task id appears in more than one split.
        task_ids_per_split = {
            split: {
                json.loads(line)["messages"][1]["content"]
                for line in (gen_dir / f"{split}.jsonl").read_text().splitlines()
                if line
            }
            for split in ("train", "valid", "test")
        }
        # Pairwise disjoint (input_text encodes the task identity).
        a, b, c = (
            task_ids_per_split["train"],
            task_ids_per_split["valid"],
            task_ids_per_split["test"],
        )
        assert a.isdisjoint(b)
        assert a.isdisjoint(c)
        assert b.isdisjoint(c)

        # 6. Stand up the trainer with a recording runner.
        runner = _RecordingRunner(tmp_path / "runner")
        backend = MLXTrainerBackend(runner=runner)

        # 7. Train, fuse, export GGUF — pointing output_dir somewhere
        #    explicitly OUTSIDE the dataset directory.
        train_out = tmp_path / "train_out"
        config = TrainConfig(
            model_id="qwen-3b",
            dataset_dir=str(gen_dir),
            output_dir=str(train_out),
            epochs=1,
            iterations=10,
            learning_rate=2e-4,
        )
        train_result = backend.train_sft(gen_dir, config)
        export = backend.merge_or_fuse(train_result)
        gguf_path = backend.export_gguf(export, "q4_0")

        # Runner methods were called, in order, with the right arguments.
        method_seq = [c[0] for c in runner.calls]
        assert method_seq == ["train_lora", "fuse_model", "convert_gguf"]
        train_args = next(args for name, args in runner.calls if name == "train_lora")
        assert train_args["model_id"] == "qwen-3b"
        assert train_args["epochs"] == 1
        assert train_args["learning_rate"] == pytest.approx(2e-4)
        assert train_args["output_dir"] == str(train_out)
        gguf_args = next(args for name, args in runner.calls if name == "convert_gguf")
        assert gguf_args["quant"] == "q4_0"

        # Runner outputs propagated into the backend artifacts.
        assert train_result.metrics == {"loss": 0.123, "step": 1000}
        assert train_result.adapter_dir == str(tmp_path / "runner" / "fake_adapter")
        assert gguf_path.name == "model-q4_0.gguf"

        # Output never landed inside the dataset directory.
        assert not (gen_dir / "train_config.json").exists()
        assert (train_out / "train_config.json").exists()

        # 8. Generate a Modelfile pointing at the GGUF path.
        modelfile = generate_modelfile(
            str(gguf_path),
            model_name="cl-agent-qwen:gen-0001",
            chat_template=chat_template,
        )

        # 9. Modelfile structural assertions.
        assert f"FROM {gguf_path}" in modelfile

        sys_match = re.search(
            r'^SYSTEM """(.*?)"""', modelfile, re.MULTILINE | re.DOTALL
        )
        assert sys_match is not None, f"no triple-quoted SYSTEM block:\n{modelfile}"
        assert sys_match.group(1) == "You write small patches."

        tmpl_match = re.search(
            r'^TEMPLATE """(.*?)"""', modelfile, re.MULTILINE | re.DOTALL
        )
        assert tmpl_match is not None, f"no triple-quoted TEMPLATE block:\n{modelfile}"
        body = tmpl_match.group(1)
        # Uses Ollama Go-template variables, not Python placeholders.
        assert "{{ .Prompt }}" in body
        assert "{{ .Response }}" in body
        assert "{{ .System }}" in body
        assert "{content}" not in body

        # ChatML stop tokens are configured.
        assert 'PARAMETER stop "<|im_end|>"' in modelfile
        assert 'PARAMETER stop "<|im_start|>"' in modelfile

        # No double EOS marker anywhere in the generated artifact.
        assert "<|im_end|><|im_end|>" not in modelfile

        # No SYSTEM line missing the opening triple-quote (the prior bug).
        for line in modelfile.splitlines():
            if line.startswith("SYSTEM"):
                assert line.startswith('SYSTEM """'), (
                    f"SYSTEM line missing opening triple-quote: {line!r}"
                )
