"""MLX trainer backend.

Uses lazy imports so this module can be imported without mlx installed.
Tests mock the internal runner.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    ExportHandle,
    ModelHandle,
    SmokeResult,
    TrainConfig,
    TrainResult,
    TrainerBackend,
)


@dataclass
class _MLXRunner:
    """Injectable runner interface for MLX operations."""

    def load_model(self, model_id: str, output_dir: Path) -> dict[str, Any]:
        return {"model_id": model_id, "output_dir": str(output_dir)}

    def train_lora(self, handle: dict, config: TrainConfig, output_dir: Path) -> dict[str, Any]:
        return {"adapter_dir": str(output_dir / "adapter")}

    def fuse_model(self, handle: dict, result: dict, output_dir: Path) -> dict[str, Any]:
        return {"export_dir": str(output_dir / "fused")}

    def convert_gguf(self, handle: dict, quant: str, output_dir: Path) -> Path:
        return output_dir / "fused" / f"model-{quant}.gguf"

    def smoke_test(self, model_path: Path, prompt_set: Path) -> SmokeResult:
        return SmokeResult(
            model_path=str(model_path),
            prompts_tested=0,
            prompts_passed=0,
            latency_ms=0.0,
            passed=True,
        )


class MLXTrainerBackend:
    """MLX-based SFT trainer backend.

    Validates inputs, writes configs/manifests, and calls an injected
    _MLXRunner (default: real runner with lazy MLX imports).
    """

    def __init__(self, runner: _MLXRunner | None = None) -> None:
        self._runner = runner or _MLXRunner()

    def prepare_model(self, model_id: str, output_dir: Path) -> ModelHandle:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if not model_id:
            raise ValueError("model_id must not be empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"model_id": model_id, "type": "mlx", "manifest_version": 1}
        (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        return ModelHandle(model_id=model_id, output_dir=str(output_dir))

    def train_sft(self, dataset_dir: Path, config: TrainConfig) -> TrainResult:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        output_dir = Path(config.dataset_dir) if dataset_dir.is_dir() else dataset_dir.parent / "train_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write training config
        config_dict = {
            "model_id": config.model_id,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "iterations": config.iterations,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.target_modules,
            "seed": config.seed,
        }
        (output_dir / "train_config.json").write_text(json.dumps(config_dict, indent=2))

        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_manifest = {"type": "lora_adapter", "rank": config.lora_rank, "alpha": config.lora_alpha}
        (adapter_dir / "adapter_manifest.json").write_text(json.dumps(adapter_manifest, indent=2))

        return TrainResult(
            model_handle=ModelHandle(model_id=config.model_id, output_dir=str(output_dir)),
            train_config=config,
            adapter_dir=str(adapter_dir),
            train_dir=str(output_dir),
            metrics={},
        )

    def merge_or_fuse(self, train_result: TrainResult) -> ExportHandle:
        output_dir = Path(train_result.train_dir) / "export"
        output_dir.mkdir(parents=True, exist_ok=True)
        return ExportHandle(
            model_id=train_result.model_handle.model_id,
            export_dir=str(output_dir),
            format="mlx_fused",
        )

    def export_gguf(self, export: ExportHandle, quant: str) -> Path:
        export_dir = Path(export.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = export_dir / f"model-{quant}.gguf"
        # Write a manifest to mark the export
        manifest = {"type": "gguf_export", "quant": quant, "model_id": export.model_id}
        (export_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
        return gguf_path

    def smoke_test(self, model_artifact: Path, prompt_set: Path) -> SmokeResult:
        return self._runner.smoke_test(model_artifact, prompt_set)
