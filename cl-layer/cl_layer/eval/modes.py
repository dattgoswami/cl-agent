"""Evaluation mode identifiers."""

from __future__ import annotations

from typing import Literal

EvalMode = Literal["baseline", "symbolic", "search", "search_sft"]

KNOWN_MODES: list[EvalMode] = [
    "baseline",
    "symbolic",
    "search",
    "search_sft",
]
