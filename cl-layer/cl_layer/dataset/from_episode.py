"""Convert verified episodes into TrainingExamples."""

from __future__ import annotations

from cl_layer.dataset.example_schema import ExampleType, TrainingExample, make_example_id
from cl_layer.episode.schema import Episode


def episode_to_example(ep: Episode) -> TrainingExample | None:
    """Convert a verified episode to a training example.

    Prefers patch_text as the target. Falls back to tool_trace.
    Returns None if the episode is not suitable for training.
    """
    # Need a successful verification outcome for direct examples
    if ep.verification_score is not None and ep.verification_score <= 0:
        return None

    # Build input from task description + context
    parts = [ep.task_description]
    if ep.task_tags:
        parts.append(f"Tags: {', '.join(ep.task_tags)}")
    if ep.repo_id:
        parts.append(f"Repo: {ep.repo_id}")
    input_text = "\n".join(p for p in parts if p)

    # Prefer patch_text, fall back to tool_trace
    if ep.patch_text:
        target_text = ep.patch_text
    elif ep.tool_trace:
        target_text = str(ep.tool_trace)
    else:
        return None

    example_id = make_example_id(input_text, target_text)
    return TrainingExample(
        id=example_id,
        input_text=input_text,
        target_text=target_text,
        example_type=ExampleType.direct,
        source_episode_id=ep.episode_id,
        metadata={
            "task_domain": ep.task_domain,
            "task_id": ep.task_id,
            "verification_score": ep.verification_score,
            "generation_id": ep.generation_id,
        },
    )
