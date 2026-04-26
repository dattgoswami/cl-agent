"""Optional TreeQuest search backend adapter.

This is a clean stub — do not add TreeQuest as a hard dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cl_layer.search.base import Candidate, Population

if TYPE_CHECKING:
    from treequest import TreeQuestClient  # type: ignore[import-untyped]


class TreeQuestSearchBackend:
    """Optional TreeQuest-based search backend.

    Lazy import — raises ImportError if TreeQuest is not installed.
    """

    def __init__(self, client: "TreeQuestClient | None" = None) -> None:
        if client is None:
            try:
                from treequest import TreeQuestClient as _TQC

                self._client = _TQC()
            except ImportError:
                raise ImportError(
                    "treequest is not installed. Install it with: pip install treequest"
                )
        else:
            self._client = client

    def ask_tell_batch(self, task: str, candidates: list[Candidate]) -> list[Candidate]:
        """Run an ask/tell batch through TreeQuest."""
        # Stub — actual implementation depends on TreeQuest API
        for cand in candidates:
            cand.metadata["backend"] = "treequest"
        return candidates

    def get_branches(self, population: Population) -> list[str]:
        """Get branch names for a population."""
        return [c.id for c in population.candidates]
