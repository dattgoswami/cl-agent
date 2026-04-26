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

    def test_train_sft_creates_config(self):
        backend = MLXTrainerBackend()
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp) / "dataset"
            ds_path.mkdir()
            config = TrainConfig(model_id="qwen-3b", dataset_dir=str(ds_path))
            result = backend.train_sft(ds_path, config)
            assert result.adapter_dir is not None
            # output_dir follows dataset_dir when it is a directory
            train_config_path = ds_path / "train_config.json"
            assert train_config_path.exists()
            config_data = json.loads(train_config_path.read_text())
            assert config_data["model_id"] == "qwen-3b"

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
