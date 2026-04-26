"""Benchmark task schema and scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkTask:
    """A single benchmark task with its evaluation commands."""

    task_id: str
    prompt: str
    repo_path: str = ""
    split: str = ""
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    verifier_commands: list[dict] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """A collection of benchmark tasks with train/valid/test splits."""

    name: str
    tasks: list[BenchmarkTask] = field(default_factory=list)

    @property
    def train_tasks(self) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.split == "train"]

    @property
    def valid_tasks(self) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.split == "valid"]

    @property
    def test_tasks(self) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.split == "test"]
