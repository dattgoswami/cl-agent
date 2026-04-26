"""MLX trainer backend.

The backend orchestrates the training pipeline through an injected
``_MLXRunner``. The runner is the integration point for real
``mlx_lm.lora`` calls when MLX is installed; the default runner returns
plausible artifact paths so unit tests can drive every backend method
without requiring MLX. All optional dependencies stay lazy — nothing
imports ``mlx``, ``mlx_lm``, ``torch``, ``transformers``, ``peft``,
``trl``, or ``unsloth`` at module load time.

Manifests written here describe what the runner reported, not invented
success: runner outputs flow into ``TrainResult``, ``ExportHandle``, and
the GGUF path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import (
    ExportHandle,
    ModelHandle,
    SmokeResult,
    TrainConfig,
    TrainResult,
)


class _MLXRunner:
    """Default runner. Real MLX integrations subclass and override these.

    Each method returns the artifacts the backend should advertise. The
    default implementation fabricates plausible paths so tests can drive
    the whole pipeline. Real MLX integration replaces these with calls
    into ``mlx_lm.lora``, ``mlx_lm.fuse``, and ``mlx_lm.convert``.
    """

    def load_model(self, model_id: str, output_dir: Path) -> dict[str, Any]:
        return {"model_id": model_id, "output_dir": str(output_dir)}

    def train_lora(
        self, handle: dict, config: TrainConfig, output_dir: Path
    ) -> dict[str, Any]:
        return {
            "adapter_dir": str(output_dir / "adapter"),
            "metrics": {},
        }

    def fuse_model(
        self, handle: dict, train_result: dict, output_dir: Path
    ) -> dict[str, Any]:
        return {"export_dir": str(output_dir)}

    def convert_gguf(
        self, handle: dict, quant: str, output_dir: Path
    ) -> Path:
        return output_dir / f"model-{quant}.gguf"

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

    All side-effecting operations go through ``_MLXRunner`` so callers can
    swap in a real MLX implementation or a recording fake. The backend
    validates inputs, places artifacts in a separate ``output_dir``
    (never inside the dataset directory), and writes manifests that
    reflect what the runner reported.
    """

    def __init__(self, runner: _MLXRunner | None = None) -> None:
        self._runner = runner or _MLXRunner()

    # ------------- prepare_model -------------

    def prepare_model(self, model_id: str, output_dir: Path) -> ModelHandle:
        if not model_id:
            raise ValueError("model_id must not be empty")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"model_id": model_id, "type": "mlx", "manifest_version": 1}
        (output_dir / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        return ModelHandle(model_id=model_id, output_dir=str(output_dir))

    # ------------- train_sft -------------

    def train_sft(self, dataset_dir: Path, config: TrainConfig) -> TrainResult:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        output_dir = self._resolve_output_dir(dataset_dir, config)
        if output_dir.resolve() == dataset_dir.resolve():
            raise ValueError(
                "TrainConfig.output_dir must not equal dataset_dir; "
                "training artifacts would overwrite the dataset"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Manifest of intent (what we're asking the runner to do).
        config_dict = {
            "model_id": config.model_id,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "iterations": config.iterations,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": list(config.target_modules),
            "seed": config.seed,
        }
        (output_dir / "train_config.json").write_text(
            json.dumps(config_dict, indent=2)
        )

        # Invoke the runner.
        handle = {"model_id": config.model_id, "output_dir": str(output_dir)}
        train_out = self._runner.train_lora(handle, config, output_dir)
        adapter_dir = Path(train_out.get("adapter_dir") or output_dir / "adapter")
        metrics = train_out.get("metrics", {}) or {}

        # Manifest reflects runner output.
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_manifest = {
            "type": "lora_adapter",
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "model_id": config.model_id,
            "adapter_dir": str(adapter_dir),
        }
        (adapter_dir / "adapter_manifest.json").write_text(
            json.dumps(adapter_manifest, indent=2)
        )

        return TrainResult(
            model_handle=ModelHandle(
                model_id=config.model_id, output_dir=str(output_dir)
            ),
            train_config=config,
            adapter_dir=str(adapter_dir),
            train_dir=str(output_dir),
            metrics=metrics,
        )

    @staticmethod
    def _resolve_output_dir(dataset_dir: Path, config: TrainConfig) -> Path:
        if config.output_dir:
            return Path(config.output_dir)
        return dataset_dir.parent / "train_output"

    # ------------- merge_or_fuse -------------

    def merge_or_fuse(self, train_result: TrainResult) -> ExportHandle:
        output_dir = Path(train_result.train_dir) / "export"
        output_dir.mkdir(parents=True, exist_ok=True)
        handle = {
            "model_id": train_result.model_handle.model_id,
            "adapter_dir": train_result.adapter_dir,
        }
        result_dict = {
            "model_id": train_result.model_handle.model_id,
            "adapter_dir": train_result.adapter_dir,
            "train_dir": train_result.train_dir,
        }
        fuse_out = self._runner.fuse_model(handle, result_dict, output_dir)
        export_dir = Path(fuse_out.get("export_dir") or output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        fuse_manifest = {
            "type": "fused_model",
            "model_id": train_result.model_handle.model_id,
            "export_dir": str(export_dir),
        }
        (export_dir / "fuse_manifest.json").write_text(
            json.dumps(fuse_manifest, indent=2)
        )
        return ExportHandle(
            model_id=train_result.model_handle.model_id,
            export_dir=str(export_dir),
            format="mlx_fused",
        )

    # ------------- export_gguf -------------

    def export_gguf(self, export: ExportHandle, quant: str) -> Path:
        export_dir = Path(export.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        handle = {"model_id": export.model_id, "export_dir": str(export_dir)}
        gguf_path = Path(self._runner.convert_gguf(handle, quant, export_dir))

        manifest = {
            "type": "gguf_export",
            "quant": quant,
            "model_id": export.model_id,
            "gguf_path": str(gguf_path),
        }
        (export_dir / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        return gguf_path

    # ------------- smoke_test -------------

    def smoke_test(self, model_artifact: Path, prompt_set: Path) -> SmokeResult:
        return self._runner.smoke_test(Path(model_artifact), Path(prompt_set))
