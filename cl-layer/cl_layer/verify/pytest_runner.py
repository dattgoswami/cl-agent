"""Pytest verification runner."""

from __future__ import annotations

from .python_repo import PythonRepoVerifier


def make_pytest_steps(
    test_path: str = "tests/",
    extra_args: list[str] | None = None,
) -> list[dict]:
    """Build a list of verification steps for running pytest."""
    cmd = ["python", "-m", "pytest"] + (extra_args or []) + [test_path]
    return [{"name": "pytest", "command": cmd, "expected_exit_code": 0, "timeout": 120}]


class PytestRunner:
    """Run pytest as part of verification."""

    def __init__(self, test_path: str = "tests/", extra_args: list[str] | None = None) -> None:
        self._steps = make_pytest_steps(test_path, extra_args or [])

    def run(self, repo_path: str, extra_env: dict[str, str] | None = None):
        return PythonRepoVerifier(self._steps).run(repo_path, extra_env)
