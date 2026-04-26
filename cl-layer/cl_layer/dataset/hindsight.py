"""Hindsight relabeling for partial successes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cl_layer.dataset.example_schema import ExampleType, TrainingExample, make_example_id
from cl_layer.episode.schema import Episode


@dataclass
class RelabeledSubtask:
    """A mechanically verifiable subtask derived from a partial success."""

    task_description: str
    target_text: str
    confidence: float  # 0.0-1.0 how confident we are this is valid


def relabel_partial_success(ep: Episode) -> list[RelabeledSubtask]:
    """Derive smaller valid tasks from partial successes.

    Rules:
    - Patch made one failing test pass -> relabel to that specific fix
    - Patch fixed import/syntax but not all tests -> relabel to the import fix
    - Patch improved lint/type errors -> relabel to the lint fix
    - Patch completed one file-local refactor correctly -> relabel to subtask
    - Patch generated correct migration but missed downstream -> relabel to migration
    """
    results: list[RelabeledSubtask] = []

    if not ep.patch_text:
        return results

    # Rule: test made it pass (from test_trace)
    if ep.test_trace:
        for trace_entry in ep.test_trace:
            if isinstance(trace_entry, dict):
                test_name = trace_entry.get("test", "")
                status = trace_entry.get("status", "")
                if "pass" in status.lower() and test_name:
                    results.append(
                        RelabeledSubtask(
                            task_description=f"Fix the failing test: {test_name}",
                            target_text=ep.patch_text,
                            confidence=0.9,
                        )
                    )

    # Rule: fixed import/syntax error
    if ep.stderr_excerpt:
        import_match = re.search(r"ImportError:\s+(\S+)", ep.stderr_excerpt)
        if import_match:
            results.append(
                RelabeledSubtask(
                    task_description=f"Fix missing import: {import_match.group(1)}",
                    target_text=ep.patch_text,
                    confidence=0.85,
                )
            )

    # Rule: improved lint errors
    lint_failures = ep.verification_failures or []
    lint_keywords = ["lint", "flake8", "ruff", "pylint"]
    for failure in lint_failures:
        if any(kw in failure.lower() for kw in lint_keywords):
            results.append(
                RelabeledSubtask(
                    task_description=f"Fix lint error mentioned in verifier output",
                    target_text=ep.patch_text,
                    confidence=0.7,
                )
            )
            break

    # Rule: partial file refactor (one file changed correctly)
    if ep.verification_score is not None and 0 < ep.verification_score < 1:
        results.append(
            RelabeledSubtask(
                task_description=f"Complete the partial task: {ep.task_description}",
                target_text=ep.patch_text,
                confidence=0.5,
            )
        )

    return results


def relabeled_to_examples(ep: Episode) -> list[TrainingExample]:
    """Convert relabeled subtasks to TrainingExamples."""
    subtasks = relabel_partial_success(ep)
    examples: list[TrainingExample] = []
    for subtask in subtasks:
        example_id = make_example_id(subtask.task_description, subtask.target_text)
        examples.append(
            TrainingExample(
                id=example_id,
                input_text=subtask.task_description,
                target_text=subtask.target_text,
                example_type=ExampleType.subtask,
                source_episode_id=ep.episode_id,
                metadata={
                    "confidence": subtask.confidence,
                    "task_domain": ep.task_domain,
                    "task_id": ep.task_id,
                },
            )
        )
    return examples
