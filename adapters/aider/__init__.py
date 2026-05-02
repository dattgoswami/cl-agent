"""Aider adapter for normalized CL episodes."""

from .context_builder import AiderRunContext, ContextBuilder
from .item_mapper import AiderMappingResult, map_aider_run
from .log_loader import AiderChatMessage, load_chat_history, load_chat_history_lines
from .runner import (
    AiderCommandResult,
    AiderProcessResult,
    AiderRunner,
    GitSnapshot,
    parse_git_status_paths,
)

__all__ = [
    "AiderChatMessage",
    "AiderCommandResult",
    "AiderMappingResult",
    "AiderProcessResult",
    "AiderRunContext",
    "AiderRunner",
    "ContextBuilder",
    "GitSnapshot",
    "load_chat_history",
    "load_chat_history_lines",
    "map_aider_run",
    "parse_git_status_paths",
]
