"""Lint verification runners (ruff, flake8)."""

from __future__ import annotations

from .python_repo import PythonRepoVerifier


def make_lint_steps(
    tool: str = "ruff",
    targets: list[str] | None = None,
) -> list[dict]:
    """Build lint verification steps."""
    targets = targets or ["."]
    if tool == "ruff":
        cmd = ["ruff", "check"] + targets
    elif tool == "flake8":
        cmd = ["flake8"] + targets
    else:
        raise ValueError(f"Unknown lint tool: {tool}")
    return [{"name": f"lint_{tool}", "command": cmd, "expected_exit_code": 0, "timeout": 60}]


class LintRunner:
    """Run lint checks as part of verification."""

    def __init__(self, tool: str = "ruff", targets: list[str] | None = None) -> None:
        self._steps = make_lint_steps(tool, targets)

    def run(self, repo_path: str, extra_env: dict[str, str] | None = None):
        return PythonRepoVerifier(self._steps).run(repo_path, extra_env)
