from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class HermesRunContext:
    task_prompt: str
    mode: Literal["baseline", "integrated"]
    cwd: str | None
    ephemeral_system_prompt: str | None
    skip_memory: bool
    skip_context_files: bool
    persist_session: bool

    def agent_kwargs(self) -> dict:
        """
        Return AIAgent keyword arguments implied by the CL operating mode.

        The adapter does not instantiate Hermes in tests.  These kwargs match
        the stable `AIAgent` constructor switches observed in Hermes:
        `ephemeral_system_prompt`, `skip_memory`, `skip_context_files`, and
        `persist_session`.
        """
        return {
            "ephemeral_system_prompt": self.ephemeral_system_prompt,
            "skip_memory": self.skip_memory,
            "skip_context_files": self.skip_context_files,
            "persist_session": self.persist_session,
        }


class ContextBuilder:
    """
    Builds Hermes run context without writing Hermes config or `~/.hermes`.

    Baseline mode disables Hermes native memory/context-file persistence through
    `AIAgent` switches when used by the injectable runner.  Integrated mode
    injects CL artifacts through Hermes' real ephemeral system-prompt mechanism.
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
    ) -> HermesRunContext:
        if mode not in {"baseline", "integrated"}:
            raise ValueError("mode must be 'baseline' or 'integrated'")

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

        return HermesRunContext(
            task_prompt=task_prompt,
            mode=mode,
            cwd=cwd,
            ephemeral_system_prompt="\n\n---\n\n".join(parts) if parts else None,
            skip_memory=mode == "baseline",
            skip_context_files=mode == "baseline",
            persist_session=mode == "integrated",
        )
