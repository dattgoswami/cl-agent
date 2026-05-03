"""
Context builder for OpenHands CL adapter.

Injection mechanism
-------------------
OpenHands reads custom microagent instructions from:

    <repo_root>/.openhands/microagents/     (per-task or global custom agents)
    <repo_root>/.openhands/skills/          (shared skills / patterns)

In ``integrated`` mode this builder writes namespaced copies of ``PROGRAM.md``
and ``SKILLS.md`` into ``.openhands/skills/`` under the configured base
directory so that an OpenHands run started afterwards picks them up
automatically without clobbering user-maintained OpenHands skill files.

In ``baseline`` mode nothing is written and no CL artifacts are injected —
the substrate is the only deliberate cross-session learning mechanism.

Attribution note
-----------------
OpenHands has its own native memory/state system.  Attribution of improvements
to CL artifacts versus native OpenHands memory requires careful experiment
design (e.g. isolated persistence dirs, ephemeral sessions).  Document this
when comparing baseline vs integrated runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PROGRAM_INJECT_FILENAME = "cl-program.md"
SKILLS_INJECT_FILENAME = "cl-skills.md"


@dataclass
class OpenHandsRunContext:
    """Context for a single OpenHands task run."""

    task_prompt: str
    mode: Literal["baseline", "integrated"]
    cwd: str | None
    injected_artifacts: list[str] = field(default_factory=list)
    skill_dir: str | None = None


class ContextBuilder:
    """
    Build an :class:`OpenHandsRunContext` and optionally inject CL artifacts.

    Parameters
    ----------
    artifacts_dir:
        Directory containing ``PROGRAM.md`` and/or ``SKILLS.md``.  Pass
        ``None`` to disable artifact injection entirely.
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
        inject_dir: str | Path | None = None,
        include_program: bool = True,
        include_skills: bool = True,
    ) -> OpenHandsRunContext:
        """
        Build a run context.

        In ``integrated`` mode, PROGRAM.md and SKILLS.md are copied to
        namespaced files under ``<inject_dir>/.openhands/skills/`` (falling
        back to *cwd* when *inject_dir* is not given).  The list of files
        written is recorded in :attr:`OpenHandsRunContext.injected_artifacts`.

        In ``baseline`` mode the artifacts are never written.

        Parameters
        ----------
        task_prompt:
            The task/issue description to pass to the agent.
        mode:
            ``"baseline"`` or ``"integrated"``.
        cwd:
            Working directory for the run (repo root).  Used as the
            injection base when *inject_dir* is not provided.
        inject_dir:
            Explicit base directory for injection.  Overrides *cwd* for
            artifact placement.
        include_program:
            Whether to inject ``PROGRAM.md`` (integrated mode only).
        include_skills:
            Whether to inject ``SKILLS.md`` (integrated mode only).
        """
        injected: list[str] = []
        resolved_skill_dir: str | None = None

        if mode == "integrated":
            base = Path(inject_dir) if inject_dir else (Path(cwd) if cwd else None)
            if base is not None:
                skills_dir = base / ".openhands" / "skills"
                skills_dir.mkdir(parents=True, exist_ok=True)
                resolved_skill_dir = str(skills_dir)

                if include_program:
                    content = self._read("PROGRAM.md")
                    if content:
                        (skills_dir / PROGRAM_INJECT_FILENAME).write_text(
                            content, encoding="utf-8"
                        )
                        injected.append(PROGRAM_INJECT_FILENAME)

                if include_skills:
                    content = self._read("SKILLS.md")
                    if content:
                        (skills_dir / SKILLS_INJECT_FILENAME).write_text(
                            content, encoding="utf-8"
                        )
                        injected.append(SKILLS_INJECT_FILENAME)

        return OpenHandsRunContext(
            task_prompt=task_prompt,
            mode=mode,
            cwd=cwd,
            injected_artifacts=injected,
            skill_dir=resolved_skill_dir,
        )
