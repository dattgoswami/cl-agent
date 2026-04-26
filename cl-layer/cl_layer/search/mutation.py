"""Mutation operators for search candidates."""

from __future__ import annotations

from cl_layer.search.base import Candidate


def prompt_mutation(candidate: Candidate, model_client, new_instruction: str) -> Candidate:
    """Regenerate candidate using prompt mutation."""
    new_plan = model_client.generate(f"Modify this plan: {candidate.plan_text}\nNew instruction: {new_instruction}")
    candidate.plan_text = new_plan
    candidate.generation += 1
    return candidate


def file_scope_mutation(candidate: Candidate, new_files: list[str]) -> Candidate:
    """Restrict candidate to a specific file scope."""
    candidate.affected_files = new_files
    candidate.generation += 1
    return candidate


def patch_hunk_mutation(candidate: Candidate, model_client, mutation_instruction: str) -> Candidate:
    """Mutate specific hunks of a patch."""
    new_patch = model_client.generate(
        f"Mutate this patch:\n{candidate.patch_text}\n{mutation_instruction}"
    )
    candidate.patch_text = new_patch
    candidate.generation += 1
    return candidate


def verifier_targeted_mutation(
    candidate: Candidate,
    failures: list[str],
    model_client,
) -> Candidate:
    """Mutate patch to specifically address verifier failures."""
    failure_summary = "\n".join(failures)
    new_patch = model_client.generate(
        f"Fix these verifier failures:\n{failure_summary}\nCurrent patch:\n{candidate.patch_text}"
    )
    candidate.patch_text = new_patch
    candidate.generation += 1
    return candidate
