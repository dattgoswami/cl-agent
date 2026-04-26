"""Deduplication by patch hash and normalized patch text."""

from __future__ import annotations

import hashlib
import re
from cl_layer.dataset.example_schema import TrainingExample


def normalize_patch(text: str) -> str:
    """Normalize patch text for comparison.

    Strips whitespace, removes file paths, and removes local variable names
    that are unlikely to affect intent.
    """
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove file path markers (lines starting with --- or +++)
    lines = [l for l in text.split("\n") if not re.match(r"^(---|\+\+\+)\s+", l)]
    text = "\n".join(lines)
    return text.strip()


def dedup_by_patch_hash(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Remove examples with duplicate patch_text."""
    seen_hashes: set[str] = set()
    deduped: list[TrainingExample] = []
    for ex in examples:
        if ex.metadata.get("patch_hash"):
            h = ex.metadata["patch_hash"]
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(ex)
        else:
            deduped.append(ex)
    return deduped


def dedup_by_normalized_text(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Remove examples with duplicate normalized target text."""
    seen_texts: set[str] = set()
    deduped: list[TrainingExample] = []
    for ex in examples:
        norm = normalize_patch(ex.target_text)
        h = hashlib.sha256(norm.encode()).hexdigest()[:16]
        if h not in seen_texts:
            seen_texts.add(h)
            deduped.append(ex)
    return deduped


def dedup_examples(examples: list[TrainingExample]) -> tuple[list[TrainingExample], dict]:
    """Run all dedup strategies. Returns (deduped list, counts)."""
    counts = {
        "input": len(examples),
        "after_patch_hash": 0,
        "after_normalized_text": 0,
    }
    after_hash = dedup_by_patch_hash(examples)
    counts["after_patch_hash"] = len(after_hash)
    after_norm = dedup_by_normalized_text(after_hash)
    counts["after_normalized_text"] = len(after_norm)
    return after_norm, counts
