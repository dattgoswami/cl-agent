"""Trainer backend protocol and core dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ModelHandle:
    """Reference to a loaded/prepared model."""

    model_id: str
    output_dir: str


@dataclass
class TrainConfig:
    """SFT training configuration."""

    model_id: str
    dataset_dir: str
    epochs: int = 3
    batch_size: int = 4
    iterations: int = 1000
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    seed: int = 42


@dataclass
class TrainResult:
    """Result of a training run."""

    model_handle: ModelHandle
    train_config: TrainConfig
    adapter_dir: str
    train_dir: str
    metrics: dict = field(default_factory=dict)


@dataclass
class ExportHandle:
    """Reference to a model export artifact."""

    model_id: str
    export_dir: str
    format: str = ""


@dataclass
class SmokeResult:
    """Result of a model smoke test."""

    model_path: str
    prompts_tested: int
    prompts_passed: int
    latency_ms: float
    passed: bool


class TrainerBackend(Protocol):
    """Protocol for SFT trainer backends."""

    def prepare_model(self, model_id: str, output_dir: Path) -> ModelHandle: ...

    def train_sft(self, dataset_dir: Path, config: TrainConfig) -> TrainResult: ...

    def merge_or_fuse(self, train_result: TrainResult) -> ExportHandle: ...

    def export_gguf(self, export: ExportHandle, quant: str) -> Path: ...

    def smoke_test(self, model_artifact: Path, prompt_set: Path) -> SmokeResult: ...
