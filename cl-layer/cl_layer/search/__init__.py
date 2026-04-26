"""SOAR-style search and repair package."""

from cl_layer.search.archive import ArchiveKey, NoveltyArchive
from cl_layer.search.base import Candidate, Population, SearchConfig
from cl_layer.search.controller import soar_loop
from cl_layer.search.crossover import (
    check_disjoint_files,
    check_non_overlapping_hunks,
    crossover,
)
from cl_layer.search.mutation import (
    file_scope_mutation,
    patch_hunk_mutation,
    prompt_mutation,
    verifier_targeted_mutation,
)
from cl_layer.search.repair import RepairPromptGenerator
from cl_layer.search.sampler import ModelClient, generate_candidates, generate_from_plan
from cl_layer.search.sandbox import AppliedCandidate, InMemorySandbox, Sandbox
from cl_layer.search.selection import (
    keep_top,
    rank_candidates,
    rank_candidates_with_scores,
    score_candidate,
    score_from_candidate,
)

__all__ = [
    "ArchiveKey",
    "NoveltyArchive",
    "Candidate",
    "Population",
    "SearchConfig",
    "soar_loop",
    "check_disjoint_files",
    "check_non_overlapping_hunks",
    "crossover",
    "file_scope_mutation",
    "patch_hunk_mutation",
    "prompt_mutation",
    "verifier_targeted_mutation",
    "RepairPromptGenerator",
    "ModelClient",
    "generate_candidates",
    "generate_from_plan",
    "AppliedCandidate",
    "InMemorySandbox",
    "Sandbox",
    "keep_top",
    "rank_candidates",
    "rank_candidates_with_scores",
    "score_candidate",
    "score_from_candidate",
]
