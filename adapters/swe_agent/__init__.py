"""SWE-agent adapter for normalized CL episodes."""

from .context_builder import ContextBuilder, SWEAgentRunContext
from .item_mapper import SWEAgentMappingResult, map_swe_agent_trajectory, paths_from_unified_diff
from .runner import SWEAgentCommandPreview, SWEAgentProcessResult, SWEAgentRunner, SWEAgentRunResult
from .trajectory_loader import (
    SWEAgentTrajectory,
    append_trajectory_episode,
    append_trajectory_episodes,
    import_trajectory,
    load_trajectory,
    load_trajectory_json,
    parse_swe_agent_datetime,
    trajectory_to_episode,
)

__all__ = [
    "ContextBuilder",
    "SWEAgentCommandPreview",
    "SWEAgentMappingResult",
    "SWEAgentProcessResult",
    "SWEAgentRunContext",
    "SWEAgentRunResult",
    "SWEAgentRunner",
    "SWEAgentTrajectory",
    "append_trajectory_episode",
    "append_trajectory_episodes",
    "import_trajectory",
    "load_trajectory",
    "load_trajectory_json",
    "map_swe_agent_trajectory",
    "parse_swe_agent_datetime",
    "paths_from_unified_diff",
    "trajectory_to_episode",
]
