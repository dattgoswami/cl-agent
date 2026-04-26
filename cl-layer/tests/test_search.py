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

    def test_repair_candidate(self):
        client = _MockClient()
        gen = RepairPromptGenerator(client)
        cand = _make_candidate()
        repaired = gen.repair_candidate(cand, ["fail"], [])
        assert "[REPAIR]" in repaired.patch_text
        assert repaired.generation == 1


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

class TestController:
    def test_soar_loop(self):
        client = _MockClient()
        # Use different failures per call so the novelty archive doesn't dedup
        call_count = [0]
        class VerifierWithDeltas:
            def run(self, patch_text):
                call_count[0] += 1
                return VerificationResult(
                    task_id="test",
                    success=True,
                    score=0.8,
                    steps=[VerificationStep(name="test", command=[], cwd="", exit_code=0, duration_ms=1, success=True)],
                    failures=[f"failure-{call_count[0]}"],
                )
        verifier = VerifierWithDeltas()
        config = SearchConfig(k_candidates=3, n_elites=2)
        pop = soar_loop("Fix auth bug", client, verifier, config)
        assert pop.size >= 1
        assert pop.candidates[0].verifier_score == 0.8

    def test_soar_loop_with_failures(self):
        client = _MockClient()
        verifier = _MockVerifier(score=0.3, failures=["test failed", "lint error"])
        config = SearchConfig(k_candidates=2, n_elites=1)
        pop = soar_loop("Fix auth bug", client, verifier, config)
        assert pop.size >= 1
