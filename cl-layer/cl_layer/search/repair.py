"""Verifier-driven repair: produce a revised candidate from failure traces."""

from __future__ import annotations

from cl_layer.search.base import Candidate

from .sampler import ModelClient


class RepairPromptGenerator:
    """Generate revised candidates from verifier failure output."""

    def __init__(self, model_client: ModelClient) -> None:
        self._client = model_client

    def generate_repair(
        self,
        candidate: Candidate,
        failures: list[str],
        step_outputs: list[str],
    ) -> str:
        """Return the model's revised patch text for a failed candidate."""
        failure_text = "\n".join(f"  - {f}" for f in failures)
        step_text = "\n".join(f"  {s[:200]}" for s in step_outputs)
        prompt = (
            "Your previous patch failed verification.\n\n"
            f"Failures:\n{failure_text}\n\n"
            f"Step outputs:\n{step_text}\n\n"
            f"Original plan:\n{candidate.plan_text}\n\n"
            f"Previous patch:\n{candidate.patch_text}\n\n"
            "Generate a revised patch that fixes all failures."
        )
        return self._client.generate(prompt)

    def repair_candidate(
        self,
        candidate: Candidate,
        failures: list[str],
        step_outputs: list[str],
    ) -> Candidate:
        """Return a NEW candidate whose ``patch_text`` is the revised patch.

        The revised patch *replaces* the old one — never appends. The new
        candidate inherits the plan and affected files, points back at the
        original via ``parent_id``, and bumps ``generation``.
        """
        revised_patch = self.generate_repair(candidate, failures, step_outputs)
        return Candidate(
            id=f"{candidate.id}-r",
            plan_text=candidate.plan_text,
            patch_text=revised_patch,
            affected_files=list(candidate.affected_files),
            verifier_score=None,
            novelty_score=0.0,
            cost_score=0.0,
            parent_id=candidate.id,
            generation=candidate.generation + 1,
            metadata={"repaired_from": candidate.id},
        )
