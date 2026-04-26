"""Convert verified episodes into TrainingExamples.

An episode becomes a direct training example only when:
- ``verification_score`` is not None and strictly greater than zero
- ``patch_text`` is non-empty (the target)

There is no fallback to ``tool_trace`` — a Python ``str(...)`` of a tool
trace is not a training-grade target. Trajectory/tool-trace style examples
are out of scope for the direct path; they belong in dedicated builders.
"""

from __future__ import annotations

from cl_layer.dataset.example_schema import ExampleType, TrainingExample, make_example_id
from cl_layer.episode.schema import Episode


def episode_to_example(ep: Episode) -> TrainingExample | None:
    """Build a direct ``TrainingExample`` from a verified, patch-bearing episode.

    Returns ``None`` if the episode is not eligible.
    """
    if ep.verification_score is None or ep.verification_score <= 0:
        return None
    if not ep.patch_text:
        return None

    parts: list[str] = [ep.task_description]
    if ep.task_tags:
        parts.append(f"Tags: {', '.join(ep.task_tags)}")
    if ep.repo_id:
        parts.append(f"Repo: {ep.repo_id}")
    input_text = "\n".join(p for p in parts if p)
    target_text = ep.patch_text

    metadata: dict = {
        "task_id": ep.task_id,
        "task_domain": ep.task_domain,
        "generation_id": ep.generation_id,
        "verification_score": ep.verification_score,
    }
    if ep.patch_hash:
        metadata["patch_hash"] = ep.patch_hash

    return TrainingExample(
        id=make_example_id(input_text, target_text),
        input_text=input_text,
        target_text=target_text,
        example_type=ExampleType.direct,
        source_episode_id=ep.episode_id,
        metadata=metadata,
    )
