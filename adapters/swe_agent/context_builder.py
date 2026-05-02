from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _jinja_raw(text: str) -> str:
    """Embed arbitrary Markdown in a Jinja-rendered SWE-agent template."""
    return "{% raw %}" + text.replace("{% endraw %}", "{% endraw %}{{ '{% endraw %}' }}{% raw %}") + "{% endraw %}"


@dataclass(frozen=True)
class SWEAgentRunContext:
    task_prompt: str
    mode: Literal["baseline", "integrated"]
    cwd: str | None
    config_overlay: dict
    injected_artifacts: list[str]

    def config_text(self) -> str:
        """
        Return the SWE-agent config overlay as JSON-formatted YAML.

        YAML accepts JSON syntax, and using stdlib JSON keeps this adapter free
        of a PyYAML runtime dependency while still writing an inspectable config
        file for `sweagent run --config`.
        """
        return json.dumps(self.config_overlay, indent=2, ensure_ascii=False) + "\n"

    def write_config(self, path: str | Path) -> Path:
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(self.config_text(), encoding="utf-8")
        return config_path


class ContextBuilder:
    """
    Builds explicit SWE-agent config overlays for CL modes.

    Baseline mode injects no CL artifacts and clears SWE-agent demonstration
    history in the overlay so replay/resume-style context does not contaminate
    controlled CL evaluation. Integrated mode injects `PROGRAM.md` and
    `SKILLS.md` through `agent.templates.strategy_template`, a native
    SWE-agent prompt mechanism appended after the instance prompt.
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
    ) -> SWEAgentRunContext:
        if mode not in {"baseline", "integrated"}:
            raise ValueError("mode must be 'baseline' or 'integrated'")

        injected: list[str] = []
        parts: list[str] = []
        if mode == "integrated":
            if include_program:
                program = self._read("PROGRAM.md")
                if program:
                    injected.append("PROGRAM.md")
                    parts.append(f"## Prior Context (PROGRAM.md)\n\n{_jinja_raw(program)}")
            if include_skills:
                skills = self._read("SKILLS.md")
                if skills:
                    injected.append("SKILLS.md")
                    parts.append(f"## Known Patterns (SKILLS.md)\n\n{_jinja_raw(skills)}")

        strategy_template = None
        if parts:
            strategy_template = (
                "CL SUBSTRATE CONTEXT\n"
                "Use this project-local context as supporting guidance. "
                "The current task remains authoritative.\n\n"
                + "\n\n---\n\n".join(parts)
            )

        templates: dict[str, object] = {
            "demonstrations": [],
            "put_demos_in_history": False,
        }
        if strategy_template is not None:
            templates["strategy_template"] = strategy_template

        overlay = {"agent": {"templates": templates}}

        return SWEAgentRunContext(
            task_prompt=task_prompt,
            mode=mode,
            cwd=cwd,
            config_overlay=overlay,
            injected_artifacts=injected,
        )
