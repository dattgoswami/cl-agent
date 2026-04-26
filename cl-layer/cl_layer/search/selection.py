"""Weighted selection scoring for search candidates."""

from __future__ import annotations

from cl_layer.search.base import Candidate


def score_candidate(
    candidate: Candidate,
    tests_fixed: int = 0,
    regressions: int = 0,
    lint_ok: bool = True,
    type_ok: bool = True,
    build_ok: bool = True,
    patch_lines: int = 0,
    max_patch_lines: int = 200,
    runtime_cost: float = 0.0,
    max_runtime: float = 1.0,
    novelty: float = 0.0,
) -> float:
    """Compute weighted selection score matching the spec.

    Weights:
    - task_success: 0.30 (based on verifier_score)
    - tests_fixed: 0.20
    - regressions: -0.15
    - lint/type/build: 0.15
    - patch_size_penalty: 0.10
    - runtime_cost: 0.05
    - novelty_bonus: 0.05
    """
    # Task success (0.30)
    task_success = (candidate.verifier_score or 0.0) * 0.30

    # Tests fixed (0.20)
    tests_score = min(tests_fixed / max(1, regressions + tests_fixed), 1.0) * 0.20

    # Regressions (-0.15)
    reg_penalty = -min(regressions * 0.05, 0.15)

    # Lint/type/build (0.15)
    status_score = (lint_ok + type_ok + build_ok) / 3.0 * 0.15

    # Patch size penalty (0.10)
    if patch_lines <= max_patch_lines:
        patch_score = 0.10
    else:
        patch_score = max(0.0, 0.10 * (1.0 - (patch_lines - max_patch_lines) / max_patch_lines))

    # Runtime cost (0.05)
    runtime_score = max(0.0, 1.0 - runtime_cost / max(max_runtime, 1e-9)) * 0.05

    # Novelty bonus (0.05)
    novelty_score = novelty * 0.05

    total = task_success + tests_score + reg_penalty + status_score + patch_score + runtime_score + novelty_score
    return max(0.0, total)


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Rank candidates by their selection score (descending)."""
    scored = [(score_candidate(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def keep_top(candidates: list[Candidate], n: int) -> list[Candidate]:
    """Keep the top n candidates by score."""
    ranked = rank_candidates(candidates)
    return ranked[:n]
