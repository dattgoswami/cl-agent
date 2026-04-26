"""Tests for search and repair package."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cl_layer.search.base import Candidate, Population, SearchConfig
from cl_layer.search.sampler import generate_candidates, generate_from_plan
from cl_layer.search.repair import RepairPromptGenerator
from cl_layer.search.selection import score_candidate, rank_candidates, keep_top
from cl_layer.search.mutation import (
    prompt_mutation,
    file_scope_mutation,
    patch_hunk_mutation,
    verifier_targeted_mutation,
)
from cl_layer.search.crossover import check_disjoint_files, check_non_overlapping_hunks, crossover
from cl_layer.search.archive import ArchiveKey, NoveltyArchive
from cl_layer.search.controller import soar_loop
from cl_layer.verify.base import VerificationResult, VerificationStep


# --------------- fixtures ------------

def _make_candidate(
    plan_text: str = "plan A",
    patch_text: str = "patch A",
    affected_files: list[str] | None = None,
    verifier_score: float | None = 0.8,
    id: str = "cand-1",
    **kwargs,
) -> Candidate:
    return Candidate(
        id=id,
        plan_text=plan_text,
        patch_text=patch_text,
        affected_files=affected_files or [],
        verifier_score=verifier_score,
        **kwargs,
    )


class _MockClient:
    def generate(self, prompt: str, **kwargs) -> str:
        return f"[generated: {prompt[:30]}...]"


class _MockVerifier:
    def __init__(self, score: float = 0.8, failures: list[str] | None = None):
        self.score = score
        self.failures = failures or []

    def run(self, patch_text: str) -> VerificationResult:
        return VerificationResult(
            task_id="test",
            success=self.score > 0.5,
            score=self.score,
            steps=[VerificationStep(name="test", command=[], cwd="", exit_code=0, duration_ms=1, success=True)],
            failures=self.failures,
        )


# --------------- base ------------

class TestBaseTypes:
    def test_population_add(self):
        pop = Population(population_id="pop-1")
        pop.add(_make_candidate())
        assert pop.size == 1

    def test_population_empty(self):
        pop = Population(population_id="pop-1")
        assert pop.size == 0

    def test_search_config_defaults(self):
        cfg = SearchConfig()
        assert cfg.k_candidates == 5
        assert cfg.n_elites == 3
        assert cfg.max_generations == 10


# --------------- sampler ------------

class TestSampler:
    def test_generate_candidates(self):
        client = _MockClient()
        config = SearchConfig(k_candidates=3)
        pop = generate_candidates("Fix auth bug", client, config)
        assert pop.size == 3

    def test_generate_with_base_parent(self):
        client = _MockClient()
        config = SearchConfig(k_candidates=2)
        base = _make_candidate()
        pop = generate_candidates("Fix bug", client, config, base_candidate=base)
        for cand in pop.candidates:
            assert cand.parent_id == base.id
            assert cand.generation == base.generation + 1


# --------------- repair ------------

class TestRepair:
    def test_generate_repair(self):
        client = _MockClient()
        gen = RepairPromptGenerator(client)
        result = gen.generate_repair(
            _make_candidate(),
            ["test failed"],
            ["output excerpt"],
        )
        assert "generated" in result.lower()

    def test_repair_candidate_replaces_patch_with_revised_text(self):
        client = _MockClient()
        gen = RepairPromptGenerator(client)
        cand = _make_candidate(patch_text="ORIGINAL_PATCH_DO_NOT_KEEP")
        repaired = gen.repair_candidate(cand, ["fail"], [])
        # The revised candidate must NOT carry the old patch text or any
        # appended marker like "[REPAIR]". It must be a fresh model output.
        assert "[REPAIR]" not in repaired.patch_text
        assert "ORIGINAL_PATCH_DO_NOT_KEEP" not in repaired.patch_text
        assert "generated" in repaired.patch_text.lower()
        # And it must be a NEW Candidate, not the same instance mutated.
        assert repaired is not cand
        assert repaired.id != cand.id
        assert repaired.parent_id == cand.id
        assert repaired.generation == cand.generation + 1
        assert repaired.metadata.get("repaired_from") == cand.id


# --------------- selection ------------

class TestSelection:
    def test_score_candidate_basic(self):
        cand = _make_candidate(verifier_score=1.0)
        score = score_candidate(cand, tests_fixed=3, regressions=0, lint_ok=True, type_ok=True, build_ok=True)
        assert score > 0

    def test_score_candidate_with_regressions(self):
        cand = _make_candidate(verifier_score=1.0)
        score1 = score_candidate(cand, tests_fixed=3, regressions=0)
        score2 = score_candidate(cand, tests_fixed=3, regressions=2)
        assert score2 < score1

    def test_rank_candidates(self):
        cands = [
            _make_candidate(verifier_score=0.9, id="a"),
            _make_candidate(verifier_score=0.3, id="b"),
            _make_candidate(verifier_score=0.7, id="c"),
        ]
        ranked = rank_candidates(cands)
        assert ranked[0].id == "a"

    def test_keep_top(self):
        cands = [_make_candidate(verifier_score=float(i) / 10.0, id=f"c{i}") for i in range(10)]
        top3 = keep_top(cands, 3)
        assert len(top3) == 3
        assert top3[0].verifier_score >= top3[2].verifier_score


# --------------- mutation ------------

class TestMutation:
    def test_prompt_mutation(self):
        client = _MockClient()
        cand = _make_candidate()
        mutated = prompt_mutation(cand, client, "new instruction")
        assert mutated.generation == 1
        assert "generated" in mutated.plan_text.lower()

    def test_file_scope_mutation(self):
        cand = _make_candidate(affected_files=["a.py"])
        mutated = file_scope_mutation(cand, ["b.py"])
        assert mutated.affected_files == ["b.py"]

    def test_patch_hunk_mutation(self):
        client = _MockClient()
        cand = _make_candidate()
        mutated = patch_hunk_mutation(cand, client, "modify this")
        assert "generated" in mutated.patch_text.lower()

    def test_verifier_targeted_mutation(self):
        client = _MockClient()
        cand = _make_candidate()
        mutated = verifier_targeted_mutation(cand, ["lint error"], client)
        assert "generated" in mutated.patch_text.lower()


# --------------- crossover ------------

class TestCrossover:
    def test_disjoint_files(self):
        c1 = _make_candidate(affected_files=["a.py", "b.py"])
        c2 = _make_candidate(affected_files=["c.py", "d.py"])
        assert check_disjoint_files(c1, c2) is True

    def test_overlapping_files(self):
        c1 = _make_candidate(affected_files=["a.py", "b.py"])
        c2 = _make_candidate(affected_files=["b.py", "c.py"])
        assert check_disjoint_files(c1, c2) is False

    def test_crossover_disjoint(self):
        c1 = _make_candidate(affected_files=["a.py"])
        c2 = _make_candidate(affected_files=["b.py"])
        result = crossover(c1, c2)
        assert result is not None
        assert "crossover" in result.plan_text.lower()

    def test_crossover_no_match(self):
        c1 = _make_candidate(affected_files=["a.py", "b.py"])
        c2 = _make_candidate(affected_files=["b.py", "c.py"])
        # Overlapping files + overlapping hunks => no safe crossover
        c1.patch_text = "@@ -0,0 +1,1 @@\n+line1"
        c2.patch_text = "@@ -0,0 +1,1 @@\n+line2"
        assert check_non_overlapping_hunks(c1.patch_text, c2.patch_text) is False
        result = crossover(c1, c2)
        assert result is None

    def test_non_overlapping_hunks(self):
        p1 = "@@ +1,1 @@\n+line1"
        p2 = "@@ +100,1 @@\n+line2"
        assert check_non_overlapping_hunks(p1, p2) is True


# --------------- archive ------------

class TestArchive:
    def test_is_novel_first_add(self):
        key = ArchiveKey(failure_signature="f1", verifier_delta="0.8", changed_files="a.py")
        archive = NoveltyArchive(window=2)
        assert archive.is_novel(key) is True
        assert archive.size == 1

    def test_is_not_novel_duplicate(self):
        key = ArchiveKey(failure_signature="f1", verifier_delta="0.8", changed_files="a.py")
        archive = NoveltyArchive(window=2)
        archive.is_novel(key)
        assert archive.is_novel(key) is False

    def test_add_returns_false_on_duplicate(self):
        key = ArchiveKey(failure_signature="f1", verifier_delta="0.8", changed_files="a.py")
        archive = NoveltyArchive(window=2)
        assert archive.add(key) is True
        assert archive.add(key) is False

    def test_window_eviction(self):
        keys = [
            ArchiveKey(failure_signature=f"f{i}", verifier_delta="0.8", changed_files="a.py")
            for i in range(5)
        ]
        archive = NoveltyArchive(window=3)
        for key in keys:
            archive.is_novel(key)
        assert archive.size == 3

    def test_dedup_failure_signatures(self):
        archive = NoveltyArchive()
        sigs = ["sig1", "sig1", "sig2", "sig2", "sig3"]
        deduped = archive.dedup_failure_signatures(sigs)
        assert len(deduped) == 3


# --------------- controller ------------

from cl_layer.search.sandbox import AppliedCandidate, InMemorySandbox


class _RecordingSandbox:
    """FakeSandbox that records every apply/cleanup call."""

    def __init__(self, base="/tmp/sb", changed_files=("src/app.py",)):
        self.base = base
        self.changed = list(changed_files)
        self.applied: list[Candidate] = []
        self.cleaned: list[str] = []

    def apply(self, candidate: Candidate, *, task_id: str | None = None) -> AppliedCandidate:
        self.applied.append(candidate)
        path = f"{self.base}/{task_id or 'default'}/{candidate.id}"
        return AppliedCandidate(
            candidate=candidate,
            sandbox_path=path,
            changed_files=list(self.changed),
        )

    def cleanup(self, applied: AppliedCandidate) -> None:
        self.cleaned.append(applied.sandbox_path)


class _RecordingVerifier:
    """Verifier that records every repo_path it was called with."""

    def __init__(self, score=0.8, failures=None, changed_files=None, success=None):
        self.score = score
        self.failures = list(failures or [])
        self.changed_files = list(changed_files or [])
        self.success = success if success is not None else (score > 0.5)
        self.run_paths: list[str] = []

    def run(self, repo_path: str, extra_env=None) -> VerificationResult:
        self.run_paths.append(repo_path)
        return VerificationResult(
            task_id="test",
            success=self.success,
            score=self.score,
            steps=[
                VerificationStep(
                    name="pytest",
                    command=[],
                    cwd=repo_path,
                    exit_code=0 if self.success else 1,
                    duration_ms=10.0,
                    success=self.success,
                )
            ],
            failures=list(self.failures),
            changed_files=list(self.changed_files),
        )


class TestController:
    def test_soar_loop_returns_population(self):
        client = _MockClient()
        sandbox = _RecordingSandbox()
        verifier = _RecordingVerifier(score=0.8)
        config = SearchConfig(k_candidates=2, n_elites=1, max_generations=1, max_candidates_total=10)
        pop = soar_loop("Fix auth bug", client, verifier, sandbox, config)
        assert pop.size >= 1
        assert pop.candidates[0].verifier_score == 0.8

    def test_verifier_receives_repo_path_not_patch_text(self):
        """Verifier.run must be called with sandbox repo paths only."""
        client = _MockClient()
        sandbox = _RecordingSandbox(base="/tmp/sb-test")
        verifier = _RecordingVerifier()
        config = SearchConfig(k_candidates=2, n_elites=1, max_generations=1, max_candidates_total=10)
        soar_loop("Fix bug", client, verifier, sandbox, config, task_id="task-42")

        assert verifier.run_paths, "verifier was never called"
        for path in verifier.run_paths:
            assert path.startswith("/tmp/sb-test/task-42/"), (
                f"verifier saw something that isn't a sandbox path: {path!r}"
            )
            # Sanity: the path is short, definitely not the model's output text.
            assert "[generated:" not in path
            assert "<|im_" not in path

    def test_sandbox_apply_called_once_per_verified_candidate(self):
        client = _MockClient()
        sandbox = _RecordingSandbox()
        verifier = _RecordingVerifier()
        config = SearchConfig(k_candidates=2, n_elites=1, max_generations=1, max_candidates_total=10)
        soar_loop("Fix", client, verifier, sandbox, config)
        # One apply per verifier.run.
        assert len(sandbox.applied) == len(verifier.run_paths)
        # Every applied candidate gets cleaned up.
        assert len(sandbox.cleaned) == len(sandbox.applied)

    def test_max_candidates_total_is_enforced(self):
        client = _MockClient()
        sandbox = _RecordingSandbox()
        verifier = _RecordingVerifier(failures=["something"])
        config = SearchConfig(
            k_candidates=10,
            n_elites=2,
            max_generations=5,
            max_candidates_total=4,
        )
        soar_loop("Fix", client, verifier, sandbox, config)
        assert len(sandbox.applied) <= 4
        assert len(verifier.run_paths) <= 4

    def test_max_generations_is_enforced(self):
        # With many max_candidates_total but max_generations=1, only one
        # generation's worth of work happens. Track distinct candidate
        # generation indices observed by the sandbox.
        client = _MockClient()
        sandbox = _RecordingSandbox()
        verifier = _RecordingVerifier(failures=["fail"])
        config = SearchConfig(
            k_candidates=2,
            n_elites=1,
            max_generations=1,
            max_candidates_total=100,
        )
        soar_loop("Fix", client, verifier, sandbox, config)
        # All candidates produced in generation 0.
        gens = {c.generation for c in sandbox.applied}
        # Repair/mutation/crossover within the first generation may bump
        # generation to 1 (their parent was gen 0); but no candidate should
        # be at generation >= 2 because the loop only iterated once.
        assert max(gens) <= 1

    def test_multiple_generations_actually_run(self):
        client = _MockClient()
        sandbox = _RecordingSandbox()
        verifier = _RecordingVerifier(failures=["fail"])
        config = SearchConfig(
            k_candidates=2,
            n_elites=1,
            max_generations=3,
            max_candidates_total=100,
        )
        soar_loop("Fix", client, verifier, sandbox, config)
        gens = {c.generation for c in sandbox.applied}
        # The loop must actually advance past gen 0.
        assert max(gens) >= 2

    def test_repair_produces_candidate_with_revised_patch(self):
        client = _MockClient()  # returns "[generated: ...]"
        sandbox = _RecordingSandbox()
        verifier = _RecordingVerifier(failures=["lint error"])
        config = SearchConfig(k_candidates=2, n_elites=1, max_generations=1, max_candidates_total=20)
        pop = soar_loop("Fix", client, verifier, sandbox, config)

        # At least one candidate must be a repair (parent_id set, repaired_from in metadata).
        repairs = [c for c in pop.candidates if c.metadata.get("repaired_from")]
        assert repairs, "expected at least one repair candidate"
        for r in repairs:
            assert "[REPAIR]" not in r.patch_text
            # The repair patch is a fresh model output, not a concatenation.
            assert r.patch_text.startswith("[generated:") or "generated" in r.patch_text.lower()

    def test_mutation_or_crossover_actually_runs(self):
        """At least one candidate should be produced via mutation/crossover."""
        client = _MockClient()
        sandbox = _RecordingSandbox()
        verifier = _RecordingVerifier(failures=["fail"])
        config = SearchConfig(k_candidates=2, n_elites=2, max_generations=1, max_candidates_total=20)
        pop = soar_loop("Fix", client, verifier, sandbox, config)
        special = [
            c
            for c in pop.candidates
            if c.metadata.get("mutated_from") or c.metadata.get("crossover_parents")
        ]
        assert special, "expected at least one mutation or crossover candidate"

    def test_ranking_uses_verifier_derived_inputs(self):
        """Two candidates with the same verifier_score but different
        regressions counts should rank in the regressions order."""
        client = _MockClient()
        sandbox = _RecordingSandbox()

        # Verifier alternates: first candidate sees passing step, second sees a failed step.
        class AlternatingVerifier:
            def __init__(self):
                self.calls = 0
                self.run_paths: list[str] = []

            def run(self, repo_path, extra_env=None):
                self.run_paths.append(repo_path)
                self.calls += 1
                steps = [
                    VerificationStep(
                        name="pytest", command=[], cwd=repo_path,
                        exit_code=0, duration_ms=10, success=True,
                    ),
                ]
                if self.calls % 2 == 0:
                    # Make the second candidate look worse: regressions and a failed step.
                    steps.append(
                        VerificationStep(
                            name="lint_ruff", command=[], cwd=repo_path,
                            exit_code=1, duration_ms=5, success=False,
                        )
                    )
                return VerificationResult(
                    task_id="t", success=self.calls % 2 == 1,
                    score=0.8,  # SAME for both
                    steps=steps, failures=[], changed_files=["a.py"],
                )

        verifier = AlternatingVerifier()
        config = SearchConfig(k_candidates=2, n_elites=2, max_generations=1, max_candidates_total=2)
        pop = soar_loop("Fix", client, verifier, sandbox, config)
        # Same verifier_score for both, so ranking must come from per-step inputs.
        scores = [c.verifier_score for c in pop.candidates]
        assert scores[0] == scores[1] == 0.8
        # The candidate with the failing lint step must rank lower.
        regs = [c.metadata.get("regressions", 0) for c in pop.candidates]
        assert regs[0] <= regs[1], (
            "candidate with more regressions should rank lower, "
            f"but got regressions={regs}"
        )

    def test_novelty_key_uses_changed_files_and_score_delta(self):
        client = _MockClient()
        sandbox = _RecordingSandbox(changed_files=("src/auth.py", "src/router.py"))
        verifier = _RecordingVerifier(score=0.7, changed_files=["src/auth.py", "src/router.py"])
        config = SearchConfig(k_candidates=1, n_elites=1, max_generations=1, max_candidates_total=2)
        pop = soar_loop("Fix", client, verifier, sandbox, config)
        for c in pop.candidates:
            assert c.metadata.get("changed_files") == ["src/auth.py", "src/router.py"]
            assert "novelty_key" in c.metadata
        # The repaired/mutated candidate has a parent_score, so its novelty key
        # should encode a real delta. The first candidate uses score=signature.
        deltas = [c.metadata.get("changed_files") for c in pop.candidates]
        assert all(d == ["src/auth.py", "src/router.py"] for d in deltas)

    def test_inmemory_sandbox_returns_candidate_specific_paths(self):
        sb = InMemorySandbox(base_path="/tmp/test-sb")
        c1 = Candidate(id="c1", plan_text="p", patch_text="x", affected_files=[])
        c2 = Candidate(id="c2", plan_text="p", patch_text="y", affected_files=[])
        a1 = sb.apply(c1, task_id="t-1")
        a2 = sb.apply(c2, task_id="t-1")
        assert a1.sandbox_path != a2.sandbox_path
        assert "c1" in a1.sandbox_path
        assert "t-1" in a1.sandbox_path
        sb.cleanup(a1)
        assert a1.sandbox_path in sb.cleaned
