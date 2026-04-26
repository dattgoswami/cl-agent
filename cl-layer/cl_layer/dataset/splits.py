"""Deterministic train/valid/test split seeded by example id."""

from __future__ import annotations

from cl_layer.dataset.example_schema import TrainingExample


def split_datasets(
    examples: list[TrainingExample],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Split examples deterministically by example id hash."""
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-9

    # Sort by id for deterministic ordering
    sorted_examples = sorted(examples, key=lambda ex: ex.id)

    n = len(sorted_examples)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    train = sorted_examples[:train_end]
    valid = sorted_examples[train_end:valid_end]
    test = sorted_examples[valid_end:]

    return train, valid, test
