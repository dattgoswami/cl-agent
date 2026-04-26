"""TrainingExample schema for SFT dataset construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class ExampleType(str, Enum):
    direct = "direct"
    repair = "repair"
    subtask = "subtask"
    preference = "preference"
    trajectory = "trajectory"
    negative = "negative"


@dataclass
class TrainingExample:
    """A single supervised fine-tuning example."""

    id: str
    input_text: str
    target_text: str
    example_type: ExampleType = ExampleType.direct
    source_episode_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input_text": self.input_text,
            "target_text": self.target_text,
            "example_type": self.example_type.value,
            "source_episode_id": self.source_episode_id,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict) -> "TrainingExample":
        return TrainingExample(
            id=d["id"],
            input_text=d["input_text"],
            target_text=d["target_text"],
            example_type=ExampleType(d.get("example_type", "direct")),
            source_episode_id=d.get("source_episode_id"),
            metadata=d.get("metadata", {}),
        )


def make_example_id(input_text: str, target_text: str) -> str:
    """Generate a stable ID from input+target."""
    raw = f"{input_text}\n---\n{target_text}"
    return "ex-" + hashlib.sha256(raw.encode()).hexdigest()[:16]
