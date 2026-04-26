"""Type-check verification runner."""

from __future__ import annotations

from .python_repo import PythonRepoVerifier


def make_typecheck_steps(
    tool: str = "mypy",
    targets: list[str] | None = None,
) -> list[dict]:
    """Build type-check verification steps."""
    targets = targets or ["."]
    if tool == "mypy":
        cmd = ["mypy"] + targets
    else:
        raise ValueError(f"Unknown typecheck tool: {tool}")
    return [{"name": f"typecheck_{tool}", "command": cmd, "expected_exit_code": 0, "timeout": 120}]


class TypecheckRunner:
    """Run type-check checks as part of verification."""

    def __init__(self, tool: str = "mypy", targets: list[str] | None = None) -> None:
        self._steps = make_typecheck_steps(tool, targets)

    def run(self, repo_path: str, extra_env: dict[str, str] | None = None):
        return PythonRepoVerifier(self._steps).run(repo_path, extra_env)
