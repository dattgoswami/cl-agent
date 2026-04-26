"""Deterministic train/valid/test split that groups by ``task_id``.

Two examples sharing a ``task_id`` always land in the same split — this
prevents leakage where a model is trained on one solution to a task and
evaluated on another solution to the same task. Examples without a
``task_id`` in metadata fall back to the example id (each example becomes
its own group).

Group → split assignment is stable across runs: groups are sorted by a
SHA-256 hash of ``f"{seed}:{group_key}"`` and partitioned by index. Re-running
on the same input produces byte-identical splits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cl_layer.dataset.example_schema import TrainingExample


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: str = "cl-agent-split"


def _group_key(example: TrainingExample) -> str:
    task_id = example.metadata.get("task_id") if example.metadata else None
    return task_id or example.id


def _hash_key(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def split_datasets(
    examples: list[TrainingExample],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: str = "cl-agent-split",
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Split examples into train/valid/test, grouping by ``task_id``."""
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-9
    if not examples:
        return [], [], []

    groups: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        groups.setdefault(_group_key(ex), []).append(ex)

    sorted_keys = sorted(groups.keys(), key=lambda k: _hash_key(seed, k))
    n = len(sorted_keys)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    train_keys = set(sorted_keys[:train_end])
    valid_keys = set(sorted_keys[train_end:valid_end])

    train: list[TrainingExample] = []
    valid: list[TrainingExample] = []
    test: list[TrainingExample] = []
    for ex in examples:
        k = _group_key(ex)
        if k in train_keys:
            train.append(ex)
        elif k in valid_keys:
            valid.append(ex)
        else:
            test.append(ex)
    return train, valid, test


def split_with_config(
    examples: list[TrainingExample], config: SplitConfig
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    return split_datasets(
        examples,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
