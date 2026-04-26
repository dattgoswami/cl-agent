"""Benchmark task schema and loader.

Three task categories are recognized:

- ``repo_local``: tasks against a curated repo we own (the primary holdout
  for promotion decisions).
- ``external_slice``: a small slice of an external benchmark such as
  SWE-bench-style issue fixes.
- ``synthetic_repair``: verifier-derived patch repair tasks generated
  from prior episodes.

Suites load from JSON via :meth:`BenchmarkSuite.from_path`. YAML is
supported as an optional convenience only if PyYAML is installed —
PyYAML is never imported at module load time and is not a required
dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

BenchmarkCategory = Literal["repo_local", "external_slice", "synthetic_repair"]
KNOWN_CATEGORIES: tuple[BenchmarkCategory, ...] = (
    "repo_local",
    "external_slice",
    "synthetic_repair",
)


class BenchmarkLoadError(ValueError):
    """Raised when a benchmark suite file is malformed."""


@dataclass
class BenchmarkTask:
    """A single benchmark task with its evaluation commands."""

    task_id: str
    prompt: str
    category: BenchmarkCategory = "repo_local"
    repo_path: str = ""
    split: str = ""
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    verifier_commands: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


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

    def filter_split(self, split: str | None) -> list[BenchmarkTask]:
        if split is None:
            return list(self.tasks)
        return [t for t in self.tasks if t.split == split]

    def filter_category(self, category: BenchmarkCategory) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.category == category]

    # ------------- Loading -------------

    @classmethod
    def from_path(cls, path: str | Path) -> "BenchmarkSuite":
        """Load a benchmark suite from a JSON or YAML file.

        YAML is supported only when PyYAML is installed; the import is
        lazy and raises a clear :class:`ImportError` if missing.
        """
        p = Path(path)
        if not p.exists():
            raise BenchmarkLoadError(f"Benchmark file not found: {p}")
        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as e:
                raise ImportError(
                    "PyYAML is required for YAML benchmark files. "
                    "Install it via: pip install pyyaml"
                ) from e
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as e:  # type: ignore[attr-defined]
                raise BenchmarkLoadError(f"Invalid YAML in {p}: {e}") from e
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise BenchmarkLoadError(f"Invalid JSON in {p}: {e}") from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Any) -> "BenchmarkSuite":
        """Build a suite from a parsed dict (validated)."""
        if not isinstance(data, dict):
            raise BenchmarkLoadError(
                f"Benchmark suite must be a dict, got {type(data).__name__}"
            )
        if "name" not in data:
            raise BenchmarkLoadError("Benchmark suite missing required field 'name'")
        if "tasks" not in data:
            raise BenchmarkLoadError("Benchmark suite missing required field 'tasks'")
        if not isinstance(data["tasks"], list):
            raise BenchmarkLoadError("'tasks' must be a list")

        tasks: list[BenchmarkTask] = []
        for idx, td in enumerate(data["tasks"]):
            if not isinstance(td, dict):
                raise BenchmarkLoadError(f"tasks[{idx}] must be a dict")
            for required in ("task_id", "prompt"):
                if required not in td:
                    raise BenchmarkLoadError(
                        f"tasks[{idx}] missing required field {required!r}"
                    )
            category = td.get("category", "repo_local")
            if category not in KNOWN_CATEGORIES:
                raise BenchmarkLoadError(
                    f"tasks[{idx}] invalid category {category!r}; "
                    f"expected one of {KNOWN_CATEGORIES}"
                )
            tasks.append(
                BenchmarkTask(
                    task_id=td["task_id"],
                    prompt=td["prompt"],
                    category=category,
                    repo_path=td.get("repo_path", ""),
                    split=td.get("split", ""),
                    domain=td.get("domain", ""),
                    tags=list(td.get("tags", [])),
                    verifier_commands=list(td.get("verifier_commands", [])),
                )
            )
        return cls(name=data["name"], tasks=tasks)
