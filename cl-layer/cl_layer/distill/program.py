from __future__ import annotations

from datetime import datetime

from .skills import SkillCandidate


def render_program_md(
    domain: str,
    skills: list[SkillCandidate],
    failure_warnings: list[str],
    generated_at: datetime | None = None,
) -> str:
    ts = (generated_at or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# PROGRAM\n\n",
        f"_Generated {ts} by the CL substrate._\n\n",
        "## Current Domain\n\n",
        f"{domain}\n\n",
    ]
    if skills:
        lines += [
            "## High-Value Patterns\n\n",
            *[f"- **{sk.domain}:** {sk.pattern_summary[:120]}\n" for sk in skills[:5]],
            "\n",
        ]
    if failure_warnings:
        lines += [
            "## Failure Warnings\n\n",
            *[f"- {w}\n" for w in failure_warnings],
            "\n",
        ]
    lines += [
        "## References\n\n",
        "- See `SKILLS.md` for full skill list.\n",
        "- See `DREAMS.md` for session history.\n",
    ]
    return "".join(lines)
