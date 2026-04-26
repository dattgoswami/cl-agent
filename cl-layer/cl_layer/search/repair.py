"""Repair prompts using verifier failure traces."""

from __future__ import annotations

from typing import Protocol

from cl_layer.search.base import Candidate

from .sampler import ModelClient


class RepairPromptGenerator:
    """Generate repair prompts from verifier failures."""

    def __init__(self, model_client: ModelClient) -> None:
        self._client = model_client

    def generate_repair(self, candidate: Candidate, failures: list[str], step_outputs: list[str]) -> str:
        """Generate a repair prompt for a failed candidate."""
        failure_text = "\n".join(f"  - {f}" for f in failures)
        step_text = "\n".join(f"  {s[:200]}" for s in step_outputs)
        prompt = (
            f"Your previous candidate failed verification.\n\n"
            f"Failures:\n{failure_text}\n\n"
            f"Step outputs:\n{step_text}\n\n"
            f"Patch to repair:\n{candidate.patch_text}\n\n"
            f"Generate a repair plan that fixes all failures."
        )
        return self._client.generate(prompt)

    def repair_candidate(self, candidate: Candidate, failures: list[str], step_outputs: list[str]) -> Candidate:
        """Repair a candidate in-place and return it."""
        repair_text = self.generate_repair(candidate, failures, step_outputs)
        candidate.patch_text += f"\n\n[REPAIR]\n{repair_text}"
        candidate.generation += 1
        return candidate
