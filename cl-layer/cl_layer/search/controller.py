"""SOAR loop orchestrator for search and repair."""

from __future__ import annotations

from cl_layer.search.archive import ArchiveKey, NoveltyArchive
from cl_layer.search.base import Candidate, Population, SearchConfig
from cl_layer.search.repair import RepairPromptGenerator
from cl_layer.search.selection import keep_top, rank_candidates, score_candidate
from cl_layer.search.sampler import generate_candidates


def soar_loop(
    task_prompt: str,
    model_client,
    verifier,
    config: SearchConfig | None = None,
) -> Population:
    """Main SOAR loop: sample, verify, select, repair, score, archive.

    Args:
        task_prompt: The task to solve.
        model_client: Injectable model client for generation.
        verifier: Injectable verifier with a run() method returning VerificationResult.
        config: Search configuration.
    """
    config = config or SearchConfig()
    archive = NoveltyArchive(window=config.novelty_window)
    population = Population(population_id="gen-000", task_id=task_prompt[:20])

    # 1. Sample k candidates
    candidates = generate_candidates(task_prompt, model_client, config)

    for candidate in candidates.candidates:
        # 2. Verify each candidate
        result = verifier.run(candidate.patch_text)
        candidate.verifier_score = result.score
        candidate.metadata["failures"] = result.failures

        # 3. Check novelty
        failure_sig = "|".join(sorted(result.failures)) if result.failures else "none"
        key = ArchiveKey(
            failure_signature=failure_sig,
            verifier_delta=str(result.score),
            changed_files="|".join(sorted(candidate.affected_files)),
        )
        if not archive.add(key):
            candidate.metadata["novel"] = False
            continue
        candidate.metadata["novel"] = True

        population.add(candidate)

    # 4. Select elites
    elites = keep_top(population.candidates, config.n_elites)

    # 5. Repair elites
    repair_gen = RepairPromptGenerator(model_client)
    for elite in elites:
        failures = elite.metadata.get("failures", [])
        if failures:
            repaired = repair_gen.repair_candidate(elite, failures, [])
            # Re-verify repaired candidate
            result = verifier.run(repaired.patch_text)
            repaired.verifier_score = result.score
            population.add(repaired)

    # 6. Rank all
    population.candidates = rank_candidates(population.candidates)

    return population
