from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AiderChatMessage:
    role: Literal["user", "assistant", "tool"]
    content: str
    index: int
    source: str = "aider_chat_history"


def _append_message(
    messages: list[AiderChatMessage],
    role: Literal["user", "assistant", "tool"],
    lines: list[str],
) -> list[str]:
    content = "".join(lines).strip()
    if content:
        messages.append(AiderChatMessage(role=role, content=content, index=len(messages)))
    return []


def _flush_buffers(
    messages: list[AiderChatMessage],
    assistant: list[str],
    user: list[str],
    tool: list[str],
) -> None:
    active = sum(bool(buffer) for buffer in (assistant, user, tool))
    if active > 1:
        logger.warning(
            "Aider chat history parser invariant violated; flushing %s active buffers",
            active,
        )
    _append_message(messages, "assistant", assistant)
    _append_message(messages, "user", user)
    _append_message(messages, "tool", tool)


def load_chat_history_lines(lines: Iterable[str]) -> list[AiderChatMessage]:
    """
    Parse Aider's `.aider.chat.history.md` style markdown.

    Aider writes user prompts as `#### ...`, tool output as blockquotes, and
    assistant text as ordinary markdown.  This parser intentionally keeps the
    roles coarse and does not infer edits from prose.
    """
    messages: list[AiderChatMessage] = []
    user: list[str] = []
    assistant: list[str] = []
    tool: list[str] = []

    for line in lines:
        if line.startswith("# aider chat started at"):
            continue

        if line.startswith("> "):
            assistant = _append_message(messages, "assistant", assistant)
            user = _append_message(messages, "user", user)
            tool.append(line[2:])
            continue

        if line.startswith("#### "):
            assistant = _append_message(messages, "assistant", assistant)
            tool = _append_message(messages, "tool", tool)
            user.append(line[5:])
            continue

        user = _append_message(messages, "user", user)
        tool = _append_message(messages, "tool", tool)
        assistant.append(line)

    _flush_buffers(messages, assistant, user, tool)
    return messages


def load_chat_history(path: str | Path) -> list[AiderChatMessage]:
    history_path = Path(path)
    if not history_path.exists():
        return []
    with history_path.open("r", encoding="utf-8", errors="ignore") as f:
        return load_chat_history_lines(f)
