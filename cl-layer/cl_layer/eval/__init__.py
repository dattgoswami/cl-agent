"""Evaluation package: benchmark schema, loader, and mode-comparison runner."""

from cl_layer.eval.benchmark import (
    KNOWN_CATEGORIES,
    BenchmarkCategory,
    BenchmarkLoadError,
    BenchmarkSuite,
    BenchmarkTask,
)
from cl_layer.eval.modes import KNOWN_MODES, EvalMode
from cl_layer.eval.runner import (
    ModeReport,
    ModeSolver,
    TaskAttempt,
    compare_modes,
    run_modes,
)

__all__ = [
    "BenchmarkCategory",
    "BenchmarkLoadError",
    "BenchmarkSuite",
    "BenchmarkTask",
    "KNOWN_CATEGORIES",
    "KNOWN_MODES",
    "EvalMode",
    "ModeReport",
    "ModeSolver",
    "TaskAttempt",
    "compare_modes",
    "run_modes",
]
