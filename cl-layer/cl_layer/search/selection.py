"""Weighted selection scoring for search candidates.

The controller stashes per-candidate verifier inputs in
``Candidate.metadata`` (``tests_fixed``, ``regressions``, ``lint_ok``,
``type_ok``, ``build_ok``, ``patch_lines``, ``runtime_cost``, ``novelty``).
``score_from_candidate`` reads those and applies the spec's weighted score.
``rank_candidates`` and ``keep_top`` use that path so ranking reflects the
real verifier outcome rather than constants.
"""

from __future__ import annotations

from cl_layer.search.base import Candidate

# Default cap on patch lines and runtime_ms for normalization.
_DEFAULT_MAX_PATCH_LINES = 200
_DEFAULT_MAX_RUNTIME_MS = 120_000.0


def score_candidate(
    candidate: Candidate,
    tests_fixed: int = 0,
    regressions: int = 0,
    lint_ok: bool = True,
    type_ok: bool = True,
    build_ok: bool = True,
    patch_lines: int = 0,
    max_patch_lines: int = _DEFAULT_MAX_PATCH_LINES,
    runtime_cost: float = 0.0,
    max_runtime: float = _DEFAULT_MAX_RUNTIME_MS,
    novelty: float = 0.0,
) -> float:
    """Compute the spec's weighted selection score.

    Weights:
      - task_success: 0.30 (from ``candidate.verifier_score``)
      - tests_fixed: 0.20 (proportion of fixed vs total touched)
      - regressions: -0.15 (capped)
      - lint/type/build status: 0.15 (averaged)
      - patch size: 0.10 (penalty for large patches)
      - runtime cost: 0.05 (penalty for slow runs)
      - novelty bonus: 0.05
    """
    task_success = (candidate.verifier_score or 0.0) * 0.30

    denom = max(1, regressions + tests_fixed)
    tests_score = min(tests_fixed / denom, 1.0) * 0.20

    reg_penalty = -min(regressions * 0.05, 0.15)

    status_score = (int(lint_ok) + int(type_ok) + int(build_ok)) / 3.0 * 0.15

    if patch_lines <= max_patch_lines:
        patch_score = 0.10
    else:
        patch_score = max(
            0.0,
            0.10 * (1.0 - (patch_lines - max_patch_lines) / max(max_patch_lines, 1)),
        )

    runtime_score = max(0.0, 1.0 - runtime_cost / max(max_runtime, 1e-9)) * 0.05

    novelty_score = novelty * 0.05

    total = (
        task_success
        + tests_score
        + reg_penalty
        + status_score
        + patch_score
        + runtime_score
        + novelty_score
    )
    return max(0.0, total)


def score_from_candidate(candidate: Candidate) -> float:
    """Score a candidate using inputs the controller stashed in metadata."""
    md = candidate.metadata or {}
    return score_candidate(
        candidate,
        tests_fixed=int(md.get("tests_fixed", 0)),
        regressions=int(md.get("regressions", 0)),
        lint_ok=bool(md.get("lint_ok", True)),
        type_ok=bool(md.get("type_ok", True)),
        build_ok=bool(md.get("build_ok", True)),
        patch_lines=int(md.get("patch_lines", 0)),
        runtime_cost=float(md.get("runtime_cost", 0.0)),
        novelty=float(md.get("novelty", 0.0)),
    )


def rank_candidates_with_scores(
    candidates: list[Candidate],
) -> tuple[list[Candidate], list[float]]:
    """Return candidates ranked by score (descending), plus the scores."""
    scored = [(score_from_candidate(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored], [s for s, _ in scored]


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Rank candidates by their selection score (descending)."""
    return rank_candidates_with_scores(candidates)[0]


def keep_top(candidates: list[Candidate], n: int) -> list[Candidate]:
    """Keep the top ``n`` candidates by score."""
    return rank_candidates(candidates)[:n]
