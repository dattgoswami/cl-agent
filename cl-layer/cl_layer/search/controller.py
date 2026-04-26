"""SOAR-style multi-generation search loop.

Contract:
  - The controller never mutates the user's live repo. Every candidate is
    applied through a ``Sandbox``.
  - The verifier is called with the *sandbox repo path*, never with raw
    patch text.
  - Each generation: sample plan→patch, verify, rank, repair failing
    elites, mutate the top elite, attempt crossover. New candidates seed
    the next generation when limits allow.
  - ``max_generations`` and ``max_candidates_total`` are honored — the
    loop stops as soon as either is exhausted.
  - Ranking uses verifier-derived scoring inputs (tests_fixed,
    regressions, lint/type/build status, runtime, patch size, novelty)
    stashed in candidate metadata by the controller. Constants do not
    drive ranking.
  - Novelty archive keys use real ``changed_files`` from the sandbox/
    verifier and a real score *delta* against a parent candidate when
    available (otherwise an honestly named score signature).
"""

from __future__ import annotations

from cl_layer.search.archive import ArchiveKey, NoveltyArchive
from cl_layer.search.base import Candidate, Population, SearchConfig
from cl_layer.search.crossover import crossover
from cl_layer.search.mutation import verifier_targeted_mutation
from cl_layer.search.repair import RepairPromptGenerator
from cl_layer.search.sampler import generate_candidates, generate_from_plan
from cl_layer.search.sandbox import AppliedCandidate, Sandbox
from cl_layer.search.selection import (
    keep_top,
    rank_candidates_with_scores,
)
from cl_layer.verify.base import VerificationResult
from cl_layer.verify.score import (
    score_build_status,
    score_lint_status,
    score_regressions,
    score_runtime_cost,
    score_tests_fixed,
    score_type_status,
)


def _scoping(config: SearchConfig, k: int) -> SearchConfig:
    return SearchConfig(
        k_candidates=k,
        n_elites=config.n_elites,
        max_generations=config.max_generations,
        max_candidates_total=config.max_candidates_total,
        novelty_window=config.novelty_window,
    )


def _record_scoring_inputs(candidate: Candidate, result: VerificationResult) -> None:
    """Stash verifier-derived scoring inputs and metadata on the candidate."""
    runtime_ms = sum(s.duration_ms for s in result.steps)
    patch_lines = candidate.patch_text.count("\n") if candidate.patch_text else 0
    md = candidate.metadata
    md["tests_fixed"] = score_tests_fixed(result.steps)
    md["regressions"] = score_regressions(result.steps)
    md["lint_ok"] = score_lint_status(result.steps) > 0.5
    md["type_ok"] = score_type_status(result.steps) > 0.5
    md["build_ok"] = score_build_status(result.steps) > 0.5
    md["patch_lines"] = patch_lines
    md["runtime_cost"] = runtime_ms
    md["runtime_cost_score"] = score_runtime_cost(runtime_ms)
    md["verification_failures"] = list(result.failures)


def _novelty_key(
    result: VerificationResult,
    changed_files: list[str],
    parent_score: float | None,
) -> ArchiveKey:
    failure_sig = "|".join(sorted(result.failures)) if result.failures else "none"
    if parent_score is not None:
        delta = round(result.score - parent_score, 4)
        delta_str = f"delta={delta}"
    else:
        delta_str = f"score={round(result.score, 2)}"
    cf = "|".join(sorted(changed_files)) if changed_files else "none"
    return ArchiveKey(
        failure_signature=failure_sig,
        verifier_delta=delta_str,
        changed_files=cf,
    )


def _verify(
    candidate: Candidate,
    sandbox: Sandbox,
    verifier,
    archive: NoveltyArchive,
    parent_score: float | None,
    task_id: str | None,
) -> VerificationResult:
    """Apply candidate through sandbox, run verifier on the sandbox path,
    record scoring inputs and novelty in the candidate metadata."""
    applied: AppliedCandidate = sandbox.apply(candidate, task_id=task_id)
    try:
        result = verifier.run(applied.sandbox_path)
    finally:
        sandbox.cleanup(applied)

    candidate.verifier_score = result.score

    # Resolve changed files: prefer verifier output, fall back to sandbox.
    changed_files = list(result.changed_files) if result.changed_files else list(applied.changed_files)
    candidate.affected_files = changed_files
    candidate.metadata["changed_files"] = changed_files

    _record_scoring_inputs(candidate, result)

    key = _novelty_key(result, changed_files, parent_score)
    is_novel = archive.add(key)
    candidate.metadata["novel"] = is_novel
    candidate.metadata["novelty"] = 1.0 if is_novel else 0.0
    candidate.metadata["novelty_key"] = key.to_hash()

    return result


def soar_loop(
    task_prompt: str,
    model_client,
    verifier,
    sandbox: Sandbox,
    config: SearchConfig | None = None,
    *,
    task_id: str | None = None,
) -> Population:
    """Run the SOAR loop. See module docstring for the contract.

    Returns the final ranked :class:`Population`.
    """
    config = config or SearchConfig()
    archive = NoveltyArchive(window=config.novelty_window)
    population = Population(
        population_id="pop-0",
        task_id=task_id or task_prompt[:20],
    )
    repair_gen = RepairPromptGenerator(model_client)
    total = 0
    best_score: float | None = None

    def budget_left() -> int:
        return max(0, config.max_candidates_total - total)

    for generation in range(config.max_generations):
        if budget_left() == 0:
            break

        # 1. Sample plan candidates, then convert plans to real patches.
        k = min(config.k_candidates, budget_left())
        if k == 0:
            break
        plan_pop = generate_candidates(task_prompt, model_client, _scoping(config, k))
        generate_from_plan(plan_pop, model_client)
        for cand in plan_pop.candidates:
            cand.generation = generation

        # 2. Apply each in sandbox + verify.
        for cand in plan_pop.candidates:
            if budget_left() == 0:
                break
            result = _verify(cand, sandbox, verifier, archive, best_score, task_id)
            population.add(cand)
            total += 1
            if best_score is None or result.score > best_score:
                best_score = result.score

        # 3. Pick elites by real weighted score.
        elites = keep_top(population.candidates, config.n_elites)
        if not elites:
            continue

        # 4. Repair failing elites — replaces the patch via the model.
        for elite in list(elites):
            failures = elite.metadata.get("verification_failures") or []
            if failures and budget_left() > 0:
                repaired = repair_gen.repair_candidate(elite, failures, [])
                _verify(repaired, sandbox, verifier, archive, elite.verifier_score, task_id)
                population.add(repaired)
                total += 1

        # 5. Mutate the top elite for next-gen diversity.
        # The mutation operator bumps generation itself, so seed at the
        # parent's generation and let the operator increment.
        if budget_left() > 0:
            top = elites[0]
            failures = top.metadata.get("verification_failures") or []
            seed = Candidate(
                id=f"{top.id}-mut",
                plan_text=top.plan_text,
                patch_text=top.patch_text,
                affected_files=list(top.affected_files),
                parent_id=top.id,
                generation=top.generation,
                metadata={"mutated_from": top.id},
            )
            mutated = verifier_targeted_mutation(
                seed, failures or ["maximize coverage"], model_client
            )
            _verify(mutated, sandbox, verifier, archive, top.verifier_score, task_id)
            population.add(mutated)
            total += 1

        # 6. Crossover top two elites if disjoint files / non-overlapping hunks.
        # The crossover operator already sets generation = max(parents) + 1.
        if len(elites) >= 2 and budget_left() > 0:
            xover = crossover(elites[0], elites[1])
            if xover is not None:
                xover.metadata = {
                    **xover.metadata,
                    "crossover_parents": [elites[0].id, elites[1].id],
                }
                _verify(xover, sandbox, verifier, archive, best_score, task_id)
                population.add(xover)
                total += 1

    # Final ranking by real weighted score.
    ranked, _ = rank_candidates_with_scores(population.candidates)
    population.candidates = ranked
    return population
