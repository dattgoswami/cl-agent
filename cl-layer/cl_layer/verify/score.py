"""Scoring helpers for verification outcomes."""

from __future__ import annotations

from .base import VerificationStep


def score_tests_fixed(steps: list[VerificationStep]) -> int:
    """Count the number of steps that succeeded."""
    return sum(1 for s in steps if s.success)


def score_regressions(steps: list[VerificationStep]) -> int:
    """Count the number of steps that failed (regressions)."""
    return sum(1 for s in steps if not s.success)


def score_lint_status(steps: list[VerificationStep]) -> float:
    """Return 1.0 if all lint steps passed, 0.0 otherwise."""
    lint_steps = [s for s in steps if "lint" in s.name.lower()]
    if not lint_steps:
        return 1.0
    return 1.0 if all(s.success for s in lint_steps) else 0.0


def score_type_status(steps: list[VerificationStep]) -> float:
    """Return 1.0 if all type-check steps passed, 0.0 otherwise."""
    type_steps = [s for s in steps if "type" in s.name.lower() or "mypy" in s.name.lower()]
    if not type_steps:
        return 1.0
    return 1.0 if all(s.success for s in type_steps) else 0.0


def score_build_status(steps: list[VerificationStep]) -> float:
    """Return 1.0 if all build-related steps passed, 0.0 otherwise."""
    build_steps = [s for s in steps if "build" in s.name.lower() or "install" in s.name.lower()]
    if not build_steps:
        return 1.0
    return 1.0 if all(s.success for s in build_steps) else 0.0


def score_patch_size(patch_text: str | None, max_lines: int = 200) -> float:
    """Penalize large patches. Returns 1.0 for small, 0.0 for giant."""
    if not patch_text:
        return 1.0
    lines = patch_text.count("\n")
    if lines <= max_lines:
        return 1.0
    return max(0.0, 1.0 - (lines - max_lines) / max_lines)


def score_runtime_cost(duration_ms: float, max_ms: float = 120_000) -> float:
    """Reward faster execution. Returns 1.0 if under max, 0.0 if way over."""
    if duration_ms <= 0:
        return 1.0
    if duration_ms <= max_ms:
        return 1.0
    return max(0.0, 1.0 - (duration_ms - max_ms) / max_ms)


def score_novelty_bonus(
    changed_files: list[str],
    archive_hashes: set[str] | None = None,
) -> float:
    """Reward candidates that touch files not seen before."""
    if not archive_hashes:
        return 1.0
    new_files = sum(1 for f in changed_files if f not in archive_hashes)
    total = max(len(changed_files), 1)
    return new_files / total
