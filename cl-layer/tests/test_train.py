"""Tests for trainer backend interfaces."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cl_layer.train.base import (
    ExportHandle,
    ModelHandle,
    SmokeResult,
    TrainConfig,
    TrainResult,
    TrainerBackend,
)
from cl_layer.train.mlx_backend import MLXTrainerBackend
from cl_layer.train.unsloth_backend import UnslothTrainerBackend
from cl_layer.train.registry import get_backend, list_backends, register_backend
from cl_layer.train.export import export_manifest
from cl_layer.train.promotion import PromotionGate, PromotionResult


# --------------- TrainConfig dataclass ------------

class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig(model_id="test", dataset_dir="/tmp/ds")
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.iterations == 1000
        assert config.learning_rate == pytest.approx(2e-4)
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert "q_proj" in config.target_modules


# --------------- MLXTrainerBackend ------------

class TestMLXTrainerBackend:
    def test_prepare_model_validates_model_id(self):
        backend = MLXTrainerBackend()
        with pytest.raises(ValueError):
            backend.prepare_model("", Path("/tmp/test"))

    def test_prepare_model_creates_manifest(self):
        backend = MLXTrainerBackend()
        with tempfile.TemporaryDirectory() as tmp:
            handle = backend.prepare_model("qwen-3b", Path(tmp) / "models")
            assert handle.model_id == "qwen-3b"
            manifest_path = Path(tmp) / "models" / "model_manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert manifest["type"] == "mlx"

    def test_train_sft_creates_config_in_separate_output_dir(self):
        backend = MLXTrainerBackend()
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "dataset"
            ds_path.mkdir()
            config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
            result = backend.train_sft(ds_path, config)
            assert result.adapter_dir is not None
            # The default output dir is a sibling of dataset_dir, NOT inside it.
            assert not (ds_path / "train_config.json").exists()
            train_config_path = ds_path.parent / "train_output" / "train_config.json"
            assert train_config_path.exists()
            config_data = json.loads(train_config_path.read_text())
            assert config_data["model_id"] == "qwen-3b"
            assert Path(config_data["output_dir"]) != ds_path

    def test_train_sft_requires_dataset_dir(self):
        backend = MLXTrainerBackend()
        with pytest.raises(FileNotFoundError):
            backend.train_sft(Path("/nonexistent"), TrainConfig(model_id="x", dataset_dir="/nonexistent"))

    def test_train_sft_requires_positive_lr(self):
        backend = MLXTrainerBackend()
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "dataset"
            ds_path.mkdir()
            config = TrainConfig(model_id="x", dataset_dir=str(ds_path), learning_rate=-1)
            with pytest.raises(ValueError):
                backend.train_sft(ds_path, config)

    def test_merge_or_fuse(self):
        backend = MLXTrainerBackend()
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "dataset"
            ds_path.mkdir()
            config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
            result = backend.train_sft(ds_path, config)
            export = backend.merge_or_fuse(result)
            assert export.format == "mlx_fused"
            assert export.export_dir is not None

    def test_export_gguf(self):
        backend = MLXTrainerBackend()
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "dataset"
            ds_path.mkdir()
            config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
            train_result = backend.train_sft(ds_path, config)
            export_handle = backend.merge_or_fuse(train_result)
            gguf_path = backend.export_gguf(export_handle, "q4_0")
            assert gguf_path.name == "model-q4_0.gguf"

    def test_smoke_test(self):
        backend = MLXTrainerBackend()
        with tempfile.TemporaryDirectory() as tmp:
            model_artifact = Path(tmp) / "model.gguf"
            model_artifact.touch()
            prompt_set = Path(tmp) / "prompts.txt"
            prompt_set.touch()
            result = backend.smoke_test(model_artifact, prompt_set)
            assert result.passed is True

    def test_protocol_compliance(self):
        """MLXTrainerBackend instances satisfy TrainerBackend protocol."""
        backend: TrainerBackend = MLXTrainerBackend()
        assert hasattr(backend, "prepare_model")
        assert hasattr(backend, "train_sft")
        assert hasattr(backend, "merge_or_fuse")
        assert hasattr(backend, "export_gguf")
        assert hasattr(backend, "smoke_test")


# --------------- Output dir + runner wiring ------------


class _RecordingRunner:
    """Fake runner that records every call and returns deterministic outputs.

    Drop-in replacement for ``_MLXRunner`` in tests. Each method appends to
    ``calls`` (in invocation order) and returns the values configured via
    constructor kwargs so tests can verify both the contract and the
    propagation of runner outputs into ``TrainResult`` / ``ExportHandle``.
    """

    def __init__(
        self,
        *,
        adapter_dir: str | None = None,
        train_metrics: dict | None = None,
        export_dir: str | None = None,
        gguf_path: Path | None = None,
        smoke_passed: bool = True,
    ):
        self._adapter_dir = adapter_dir
        self._train_metrics = train_metrics or {"loss": 1.234}
        self._export_dir = export_dir
        self._gguf_path = gguf_path
        self._smoke_passed = smoke_passed
        self.calls: list[tuple[str, dict]] = []

    def load_model(self, model_id, output_dir):
        self.calls.append(("load_model", {"model_id": model_id, "output_dir": str(output_dir)}))
        return {"model_id": model_id, "output_dir": str(output_dir)}

    def train_lora(self, handle, config, output_dir):
        self.calls.append(
            (
                "train_lora",
                {
                    "handle": handle,
                    "model_id": config.model_id,
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "lora_rank": config.lora_rank,
                    "output_dir": str(output_dir),
                },
            )
        )
        return {
            "adapter_dir": self._adapter_dir or str(output_dir / "adapter"),
            "metrics": self._train_metrics,
        }

    def fuse_model(self, handle, train_result, output_dir):
        self.calls.append(
            (
                "fuse_model",
                {
                    "handle": handle,
                    "train_result": train_result,
                    "output_dir": str(output_dir),
                },
            )
        )
        return {"export_dir": self._export_dir or str(output_dir)}

    def convert_gguf(self, handle, quant, output_dir):
        self.calls.append(
            (
                "convert_gguf",
                {"handle": handle, "quant": quant, "output_dir": str(output_dir)},
            )
        )
        return self._gguf_path or (output_dir / f"model-{quant}.gguf")

    def smoke_test(self, model_path, prompt_set):
        from cl_layer.train.base import SmokeResult

        self.calls.append(
            (
                "smoke_test",
                {"model_path": str(model_path), "prompt_set": str(prompt_set)},
            )
        )
        return SmokeResult(
            model_path=str(model_path),
            prompts_tested=1,
            prompts_passed=1 if self._smoke_passed else 0,
            latency_ms=42.0,
            passed=self._smoke_passed,
        )


class TestOutputDir:
    def test_explicit_output_dir_is_respected(self, tmp_path):
        ds_path = tmp_path / "dataset"
        ds_path.mkdir()
        out_path = tmp_path / "explicit_train_out"
        runner = _RecordingRunner()
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(
            model_id="qwen-3b", dataset_dir=str(ds_path), output_dir=str(out_path)
        )
        result = backend.train_sft(ds_path, config)
        # train_config.json appears under the explicit output dir.
        assert (out_path / "train_config.json").exists()
        # ...and NOT inside the dataset.
        assert not (ds_path / "train_config.json").exists()
        assert result.train_dir == str(out_path)

    def test_default_output_dir_is_outside_dataset_dir(self, tmp_path):
        ds_path = tmp_path / "dataset"
        ds_path.mkdir()
        runner = _RecordingRunner()
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
        result = backend.train_sft(ds_path, config)
        # The default sibling location is what gets used.
        expected = ds_path.parent / "train_output"
        assert Path(result.train_dir) == expected
        assert (expected / "train_config.json").exists()
        # Critically, nothing was written into the dataset.
        assert list(ds_path.iterdir()) == []

    def test_output_dir_equal_to_dataset_dir_is_rejected(self, tmp_path):
        ds_path = tmp_path / "dataset"
        ds_path.mkdir()
        backend = MLXTrainerBackend(runner=_RecordingRunner())
        config = TrainConfig(
            model_id="qwen-3b",
            dataset_dir=str(ds_path),
            output_dir=str(ds_path),
        )
        with pytest.raises(ValueError, match="output_dir"):
            backend.train_sft(ds_path, config)


class TestRunnerWiring:
    def test_train_sft_invokes_train_lora(self, tmp_path):
        ds_path = tmp_path / "ds"
        ds_path.mkdir()
        runner = _RecordingRunner(
            adapter_dir=str(tmp_path / "custom_adapter"),
            train_metrics={"loss": 0.42, "epoch": 3},
        )
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(
            model_id="qwen-3b", dataset_dir=str(ds_path), epochs=2, learning_rate=1e-4
        )
        result = backend.train_sft(ds_path, config)

        # Exactly one train_lora call, with the right config fields.
        train_calls = [c for c in runner.calls if c[0] == "train_lora"]
        assert len(train_calls) == 1
        args = train_calls[0][1]
        assert args["model_id"] == "qwen-3b"
        assert args["epochs"] == 2
        assert args["learning_rate"] == pytest.approx(1e-4)
        assert args["lora_rank"] == config.lora_rank
        # Runner output flows into TrainResult.
        assert result.adapter_dir == str(tmp_path / "custom_adapter")
        assert result.metrics == {"loss": 0.42, "epoch": 3}

    def test_merge_or_fuse_invokes_fuse_model(self, tmp_path):
        ds_path = tmp_path / "ds"
        ds_path.mkdir()
        runner = _RecordingRunner(export_dir=str(tmp_path / "custom_export"))
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
        train_result = backend.train_sft(ds_path, config)

        export = backend.merge_or_fuse(train_result)
        fuse_calls = [c for c in runner.calls if c[0] == "fuse_model"]
        assert len(fuse_calls) == 1
        args = fuse_calls[0][1]
        assert args["handle"]["model_id"] == "qwen-3b"
        assert args["handle"]["adapter_dir"] == train_result.adapter_dir
        # Runner export_dir propagates into ExportHandle.
        assert export.export_dir == str(tmp_path / "custom_export")
        assert export.format == "mlx_fused"

    def test_export_gguf_invokes_convert_gguf(self, tmp_path):
        ds_path = tmp_path / "ds"
        ds_path.mkdir()
        custom_gguf = tmp_path / "custom_dir" / "weights-q5_k.gguf"
        runner = _RecordingRunner(gguf_path=custom_gguf)
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
        train_result = backend.train_sft(ds_path, config)
        export = backend.merge_or_fuse(train_result)

        path = backend.export_gguf(export, "q5_k")
        gguf_calls = [c for c in runner.calls if c[0] == "convert_gguf"]
        assert len(gguf_calls) == 1
        assert gguf_calls[0][1]["quant"] == "q5_k"
        assert path == custom_gguf

    def test_runner_call_order(self, tmp_path):
        ds_path = tmp_path / "ds"
        ds_path.mkdir()
        runner = _RecordingRunner()
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))

        result = backend.train_sft(ds_path, config)
        export = backend.merge_or_fuse(result)
        backend.export_gguf(export, "q4_0")

        method_sequence = [c[0] for c in runner.calls]
        # The runner sees train_lora first, then fuse_model, then convert_gguf.
        assert method_sequence == ["train_lora", "fuse_model", "convert_gguf"]

    def test_invalid_learning_rate_rejects_before_runner_work(self, tmp_path):
        ds_path = tmp_path / "ds"
        ds_path.mkdir()
        runner = _RecordingRunner()
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(
            model_id="qwen-3b", dataset_dir=str(ds_path), learning_rate=-0.001
        )
        with pytest.raises(ValueError, match="learning_rate"):
            backend.train_sft(ds_path, config)
        # Runner was never invoked; no train artifacts written.
        assert runner.calls == []
        assert not (ds_path.parent / "train_output").exists()

    def test_smoke_test_invokes_runner(self, tmp_path):
        runner = _RecordingRunner(smoke_passed=True)
        backend = MLXTrainerBackend(runner=runner)
        artifact = tmp_path / "model.gguf"
        artifact.touch()
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("hello\n")
        result = backend.smoke_test(artifact, prompts)
        assert result.passed is True
        smoke_calls = [c for c in runner.calls if c[0] == "smoke_test"]
        assert len(smoke_calls) == 1
        assert smoke_calls[0][1]["model_path"] == str(artifact)
        assert smoke_calls[0][1]["prompt_set"] == str(prompts)

    def test_adapter_manifest_reflects_runner_adapter_dir(self, tmp_path):
        ds_path = tmp_path / "ds"
        ds_path.mkdir()
        runner = _RecordingRunner(adapter_dir=str(tmp_path / "runner_adapter"))
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
        result = backend.train_sft(ds_path, config)

        adapter_manifest = json.loads(
            (Path(result.adapter_dir) / "adapter_manifest.json").read_text()
        )
        assert adapter_manifest["adapter_dir"] == str(tmp_path / "runner_adapter")
        assert adapter_manifest["model_id"] == "qwen-3b"

    def test_export_manifest_reflects_runner_gguf_path(self, tmp_path):
        ds_path = tmp_path / "ds"
        ds_path.mkdir()
        custom_gguf = tmp_path / "custom_dir" / "model-q4_0.gguf"
        runner = _RecordingRunner(gguf_path=custom_gguf)
        backend = MLXTrainerBackend(runner=runner)
        config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
        train_result = backend.train_sft(ds_path, config)
        export = backend.merge_or_fuse(train_result)
        backend.export_gguf(export, "q4_0")

        export_manifest = json.loads(
            (Path(export.export_dir) / "export_manifest.json").read_text()
        )
        assert export_manifest["gguf_path"] == str(custom_gguf)
        assert export_manifest["quant"] == "q4_0"


# --------------- prepare_model: no duplicate validation ------------


class TestPrepareModelValidation:
    def test_empty_model_id_raises_once(self, tmp_path):
        backend = MLXTrainerBackend()
        with pytest.raises(ValueError, match="model_id"):
            backend.prepare_model("", tmp_path / "models")


# --------------- UnslothTrainerBackend ------------

class TestUnslothTrainerBackend:
    def test_all_methods_raise_not_implemented(self):
        backend = UnslothTrainerBackend()
        import tempfile
        for method_name in ["prepare_model", "train_sft", "merge_or_fuse", "export_gguf"]:
            method = getattr(backend, method_name)
            with pytest.raises(NotImplementedError, match="CUDA"):
                if method_name == "prepare_model":
                    method("x", Path("/tmp"))
                elif method_name == "train_sft":
                    method(Path("/tmp"), TrainConfig(model_id="x", dataset_dir="/tmp"))
                elif method_name == "merge_or_fuse":
                    method(
                        TrainResult(
                            model_handle=ModelHandle("x", "/tmp"),
                            train_config=TrainConfig(model_id="x", dataset_dir="/tmp"),
                            adapter_dir="/tmp",
                            train_dir="/tmp",
                        ),
                    )
                elif method_name == "export_gguf":
                    method(ExportHandle("x", "/tmp"), "q4")


# --------------- Registry ------------

class TestRegistry:
    def test_register_and_get(self):
        register_backend("test", MLXTrainerBackend)
        backend_cls = get_backend("test")
        assert backend_cls is MLXTrainerBackend

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_list_backends(self):
        register_backend("__test_reg", MLXTrainerBackend)
        backends = list_backends()
        assert "__test_reg" in backends


# --------------- Export ------------

class TestExport:
    def test_export_manifest(self):
        train = TrainResult(
            model_handle=ModelHandle("qwen-3b", "/tmp/models"),
            train_config=TrainConfig(model_id="qwen-3b", dataset_dir="/tmp"),
            adapter_dir="/tmp/adapter",
            train_dir="/tmp/train",
        )
        export = ExportHandle("qwen-3b-export", "/tmp/export", "mlx_fused")
        manifest = export_manifest(train, export)
        assert manifest["source_model"] == "qwen-3b"
        assert manifest["export_format"] == "mlx_fused"
