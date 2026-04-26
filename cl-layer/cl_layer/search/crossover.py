"""Crossover operators for combining candidates."""

from __future__ import annotations

from cl_layer.search.base import Candidate


def check_disjoint_files(c1: Candidate, c2: Candidate) -> bool:
    """Check if two candidates touch disjoint files."""
    files1 = set(c1.affected_files)
    files2 = set(c2.affected_files)
    return not files1 & files2


def check_non_overlapping_hunks(patch1: str, patch2: str) -> bool:
    """Check if two patches have non-overlapping hunks.

    Simple heuristic: check that @@ markers don't share line numbers.
    """
    import re

    hunks1 = set(re.findall(r"@@\s+[-+]\d+,\d+\s+[-+](\d+),\d+\s+@@", patch1))
    hunks2 = set(re.findall(r"@@\s+[-+]\d+,\d+\s+[-+](\d+),\d+\s+@@", patch2))
    return not hunks1 & hunks2


def crossover(c1: Candidate, c2: Candidate) -> Candidate | None:
    """Combine two candidates only when edits touch disjoint files or non-overlapping hunks."""
    # Check file disjointness
    if check_disjoint_files(c1, c2):
        return Candidate(
            id=f"{c1.id}-{c2.id}",
            plan_text=f"[crossover]\n{c1.plan_text}\n{c2.plan_text}",
            patch_text=f"[crossover files disjoint]\n{c1.patch_text}\n{c2.patch_text}",
            affected_files=list(set(c1.affected_files) | set(c2.affected_files)),
            generation=max(c1.generation, c2.generation) + 1,
            parent_id=c1.id,
            metadata={"parent2_id": c2.id, "crossover_type": "disjoint_files"},
        )

    # Check non-overlapping hunks
    if check_non_overlapping_hunks(c1.patch_text, c2.patch_text):
        return Candidate(
            id=f"{c1.id}-{c2.id}",
            plan_text=f"[crossover]\n{c1.plan_text}\n{c2.plan_text}",
            patch_text=f"[crossover non-overlapping]\n{c1.patch_text}\n{c2.patch_text}",
            affected_files=list(set(c1.affected_files) | set(c2.affected_files)),
            generation=max(c1.generation, c2.generation) + 1,
            parent_id=c1.id,
            metadata={"parent2_id": c2.id, "crossover_type": "non_overlapping"},
        )

    # Cannot safely crossover
    return None
