from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PiRunContext:
    task_prompt: str
    append_system_prompt: str | None
    cwd: str | None
    mode: Literal["baseline", "integrated"]

    def cli_args(self) -> list[str]:
        """Return Pi CLI flags implied by the CL operating mode."""
        args: list[str] = []
        if self.mode == "baseline":
            args.append("--no-session")
        if self.append_system_prompt:
            args.extend(["--append-system-prompt", self.append_system_prompt])
        return args


class ContextBuilder:
    """
    Builds Pi run context without writing Pi config.

    Pi supports appending to the real system prompt via the CLI
    `--append-system-prompt` flag and the SDK resource loader's
    append-system-prompt mechanism.  The adapter uses that route for
    integrated mode because it is explicit, temporary, and does not require
    creating `.pi/APPEND_SYSTEM.md` or user-level configuration.
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
    ) -> PiRunContext:
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

        append_system_prompt = "\n\n---\n\n".join(parts) if parts else None
        return PiRunContext(
            task_prompt=task_prompt,
            append_system_prompt=append_system_prompt,
            cwd=cwd,
            mode=mode,
        )
