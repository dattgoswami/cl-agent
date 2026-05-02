"""Hermes Agent adapter for normalized CL episodes."""

from .context_builder import ContextBuilder, HermesRunContext
from .item_mapper import HermesMappingResult, map_hermes_conversations
from .runner import HermesAgentRunner, HermesRunResult
from .trajectory_loader import (
    HermesMalformedLine,
    HermesTrajectoryBatch,
    HermesTrajectoryEntry,
    append_trajectory_episodes,
    batch_to_episodes,
    import_trajectory_jsonl,
    load_trajectory_json,
    load_trajectory_jsonl,
    load_trajectory_lines,
    trajectory_to_episode,
)

__all__ = [
    "ContextBuilder",
    "HermesAgentRunner",
    "HermesMalformedLine",
    "HermesMappingResult",
    "HermesRunContext",
    "HermesRunResult",
    "HermesTrajectoryBatch",
    "HermesTrajectoryEntry",
    "append_trajectory_episodes",
    "batch_to_episodes",
    "import_trajectory_jsonl",
    "load_trajectory_json",
    "load_trajectory_jsonl",
    "load_trajectory_lines",
    "map_hermes_conversations",
    "trajectory_to_episode",
]
