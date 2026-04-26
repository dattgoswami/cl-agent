"""Generate candidate plans/patches from a base model client."""

from __future__ import annotations

import uuid
from typing import Protocol

from cl_layer.search.base import Candidate, Population, SearchConfig


class ModelClient(Protocol):
    """Injectable model client for candidate generation."""

    def generate(self, prompt: str, **kwargs) -> str: ...


def generate_candidates(
    task_prompt: str,
    model_client: ModelClient,
    config: SearchConfig,
    base_candidate: Candidate | None = None,
) -> Population:
    """Generate ``k_candidates`` plan-only candidates.

    ``patch_text`` is left empty; call :func:`generate_from_plan` to fill it
    with a real model-generated patch before applying in a sandbox.
    """
    population = Population(
        population_id=str(uuid.uuid4())[:8],
        task_id=task_prompt[:20],
    )

    for i in range(config.k_candidates):
        prompt = f"Task: {task_prompt}\n\nGenerate candidate plan {i + 1}."
        plan_text = model_client.generate(prompt)
        candidate = Candidate(
            id=str(uuid.uuid4())[:8],
            plan_text=plan_text,
            patch_text="",
            affected_files=[],
            generation=0,
        )
        if base_candidate:
            candidate.parent_id = base_candidate.id
            candidate.generation = base_candidate.generation + 1
        population.add(candidate)

    return population


def generate_from_plan(
    population: Population, model_client: ModelClient
) -> Population:
    """Turn each candidate's plan into a real patch via the model."""
    for cand in population.candidates:
        if cand.affected_files:
            prompt = (
                f"Generate a patch for files: {', '.join(cand.affected_files)}\n"
                f"Plan:\n{cand.plan_text}"
            )
        else:
            prompt = f"Generate a patch for plan:\n{cand.plan_text}"
        cand.patch_text = model_client.generate(prompt)
    return population
