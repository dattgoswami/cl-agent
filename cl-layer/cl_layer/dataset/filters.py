"""Conservative filtering for training examples."""

from __future__ import annotations

from cl_layer.dataset.example_schema import TrainingExample


def reject_no_verifier(example: TrainingExample) -> bool:
    """Reject examples with no verifier outcome."""
    score = example.metadata.get("verification_score")
    return score is None


def reject_empty_target(example: TrainingExample) -> bool:
    """Reject examples with empty target text."""
    return not example.target_text or not example.target_text.strip()


def reject_giant_diff(example: TrainingExample, max_hunks: int = 50) -> bool:
    """Reject giant low-signal diffs (> max hunks)."""
    hunk_count = example.target_text.count("\n@@")
    return hunk_count > max_hunks


def reject_hidden_state(example: TrainingExample) -> bool:
    """Reject examples that depend on hidden state.

    Heuristic: reject if target contains session/temp file patterns.
    """
    hidden_patterns = [
        "/tmp/",
        "/var/folders/",
        "session_id",
        "random_seed",
        "uuid.uuid4",
        "os.urandom",
    ]
    for pat in hidden_patterns:
        if pat in example.target_text:
            return True
    return False


def filter_examples(examples: list[TrainingExample]) -> tuple[list[TrainingExample], dict]:
    """Apply all filtering rules. Returns (filtered list, counts)."""
    counts = {"input": len(examples)}
    filtered = examples

    for name, reject_fn in [
        ("no_verifier", reject_no_verifier),
        ("empty_target", reject_empty_target),
        ("giant_diff", reject_giant_diff),
        ("hidden_state", reject_hidden_state),
    ]:
        before = len(filtered)
        filtered = [ex for ex in filtered if not reject_fn(ex)]
        counts[f"rejected_{name}"] = before - len(filtered)

    counts["output"] = len(filtered)
    return filtered, counts
