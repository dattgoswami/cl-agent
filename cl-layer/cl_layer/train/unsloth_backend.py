"""Unsloth trainer backend stub.

On Apple Silicon this backend is not functional and raises NotImplementedError
for local execution. It serves as an interface contract for NVIDIA/Linux.
"""

from __future__ import annotations

from pathlib import Path

from .base import ExportHandle, ModelHandle, SmokeResult, TrainConfig, TrainResult, TrainerBackend


class UnslothTrainerBackend:
    """Unsloth-based SFT trainer backend.

    Raises NotImplementedError for local execution. Use on NVIDIA/Linux
    with CUDA support.
    """

    def prepare_model(self, model_id: str, output_dir: Path) -> ModelHandle:
        raise NotImplementedError("UnslothTrainerBackend requires CUDA/NVIDIA support")

    def train_sft(self, dataset_dir: Path, config: TrainConfig) -> TrainResult:
        raise NotImplementedError("UnslothTrainerBackend requires CUDA/NVIDIA support")

    def merge_or_fuse(self, train_result: TrainResult) -> ExportHandle:
        raise NotImplementedError("UnslothTrainerBackend requires CUDA/NVIDIA support")

    def export_gguf(self, export: ExportHandle, quant: str) -> Path:
        raise NotImplementedError("UnslothTrainerBackend requires CUDA/NVIDIA support")

    def smoke_test(self, model_artifact: Path, prompt_set: Path) -> SmokeResult:
        raise NotImplementedError("UnslothTrainerBackend requires CUDA/NVIDIA support")
