from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AiderRunContext:
    task_prompt: str
    message: str
    cwd: str | None
    mode: Literal["baseline", "integrated"]

    def cli_args(self) -> list[str]:
        return ["--message", self.message]


class ContextBuilder:
    """
    Builds an Aider one-shot prompt.

    Aider supports a non-interactive `--message`/`--message-file` mode.  The
    adapter uses that explicit prompt path for CL artifact injection instead of
    writing Aider config or relying on restored chat history.
    """

    def __init__(self, artifacts_dir: str | Path | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

    def _read(self, name: str) -> str | None:
        if self.artifacts_dir is None:
            return None
        path = self.artifacts_dir / name
        return path.read_text(encoding="utf-8") if path.exists() else None

    def build(
        self,
        task_prompt: str,
        mode: Literal["baseline", "integrated"],
        cwd: str | None = None,
        include_program: bool = True,
        include_skills: bool = True,
    ) -> AiderRunContext:
        parts: list[str] = []

        if mode == "integrated":
            if include_program:
                program = self._read("PROGRAM.md")
                if program:
                    parts.append(f"## Prior Context (PROGRAM.md)\n\n{program}")
            if include_skills:
                skills = self._read("SKILLS.md")
                if skills:
                    parts.append(f"## Known Patterns (SKILLS.md)\n\n{skills}")

        if parts:
            parts.append(f"## Current Task\n\n{task_prompt}")
            message = "\n\n---\n\n".join(parts)
        else:
            message = task_prompt

        return AiderRunContext(
            task_prompt=task_prompt,
            message=message,
            cwd=cwd,
            mode=mode,
        )
