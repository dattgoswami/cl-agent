from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class RunContext:
    task_prompt: str
    developer_instructions: str | None
    cwd: str | None
    mode: Literal["baseline", "integrated"]


class ContextBuilder:
    """
    Builds a RunContext by optionally injecting substrate artifacts
    (PROGRAM.md, SKILLS.md) as developer instructions.

    Artifacts are NOT assumed to be auto-consumed by Codex; this class
    must inject them explicitly through the developer_instructions field.
    """

    def __init__(self, artifacts_dir: str | Path | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

    def _read(self, name: str) -> str | None:
        if self.artifacts_dir is None:
            return None
        p = self.artifacts_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    def build(
        self,
        task_prompt: str,
        mode: Literal["baseline", "integrated"],
        cwd: str | None = None,
        include_program: bool = True,
        include_skills: bool = True,
    ) -> RunContext:
        """
        In `integrated` mode, PROGRAM.md and SKILLS.md are injected as
        developer instructions if they exist.  In `baseline` mode they
        are omitted so the substrate is the only cross-session signal.
        """
        dev_parts: list[str] = []

        if mode == "integrated":
            if include_program:
                program = self._read("PROGRAM.md")
                if program:
                    dev_parts.append(f"## Prior Context (PROGRAM.md)\n\n{program}")
            if include_skills:
                skills = self._read("SKILLS.md")
                if skills:
                    dev_parts.append(f"## Known Patterns (SKILLS.md)\n\n{skills}")

        developer_instructions = "\n\n---\n\n".join(dev_parts) if dev_parts else None

        return RunContext(
            task_prompt=task_prompt,
            developer_instructions=developer_instructions,
            cwd=cwd,
            mode=mode,
        )
