"""Tests for verifier framework and scoring."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cl_layer.verify.base import (
    CommandResult,
    CommandRunner,
    VerificationResult,
    VerificationStep,
)
from cl_layer.verify.python_repo import (
    PythonRepoVerifier,
    SubprocessRunner,
    _compute_score,
    extract_changed_files,
)
from cl_layer.verify.score import (
    score_build_status,
    score_lint_status,
    score_novelty_bonus,
    score_patch_size,
    score_regressions,
    score_runtime_cost,
    score_tests_fixed,
    score_type_status,
)


class TestVerificationStep:
    def test_defaults(self):
        step = VerificationStep(
            name="test",
            command=["pytest"],
            cwd="/tmp",
            exit_code=0,
            duration_ms=100.0,
        )
        assert step.stdout_excerpt == ""
        assert step.stderr_excerpt == ""
        assert step.success is False

    def test_success_flag(self):
        step = VerificationStep(
            name="test",
            command=["pytest"],
            cwd="/tmp",
            exit_code=0,
            duration_ms=100.0,
            success=True,
        )
        assert step.success is True


class TestPythonRepoVerifier:
    def test_run_all_steps_succeed(self):
        steps = [
            {"name": "lint", "command": ["ruff", "check"], "expected_exit_code": 0, "timeout": 10},
            {"name": "test", "command": ["python", "-m", "pytest"], "expected_exit_code": 0, "timeout": 10},
        ]
        verifier = PythonRepoVerifier(steps)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"ok\n"
        mock_result.stderr = b""
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = verifier.run("/tmp/repo")
        assert result.success is True
        assert result.score == pytest.approx(1.0)
        assert len(result.steps) == 2
        assert result.failures == []

    def test_run_step_fails(self):
        steps = [
            {"name": "lint", "command": ["ruff", "check"], "expected_exit_code": 0, "timeout": 10},
        ]
        verifier = PythonRepoVerifier(steps)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"error\n"
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = verifier.run("/tmp/repo")
        assert result.success is False
        assert result.score == pytest.approx(0.0)
        assert len(result.failures) == 1

    def test_run_with_timeout(self):
        steps = [
            {"name": "test", "command": ["python", "-m", "pytest"], "expected_exit_code": 0, "timeout": 1},
        ]
        verifier = PythonRepoVerifier(steps)
        import subprocess
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python"], timeout=1)
            result = verifier.run("/tmp/repo")
        assert result.success is False
        assert any("timeout" in f for f in result.failures)

    def test_empty_steps(self):
        verifier = PythonRepoVerifier([])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""
        mock_result.stderr = b""
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = verifier.run("/tmp/repo")
        assert result.score == pytest.approx(0.0)


class TestComputeScore:
    def test_all_pass(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert _compute_score(steps) == pytest.approx(1.0)

    def test_half_fail(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
        ]
        assert _compute_score(steps) == pytest.approx(0.5)


# --------------- scoring helpers ------------

class TestScoringHelpers:
    def test_score_tests_fixed(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
            VerificationStep(name="c", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_tests_fixed(steps) == 2

    def test_score_regressions(self):
        steps = [
            VerificationStep(name="a", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
            VerificationStep(name="b", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
        ]
        assert score_regressions(steps) == 1

    def test_score_lint_status_all_pass(self):
        steps = [
            VerificationStep(name="lint_ruff", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_lint_status(steps) == 1.0

    def test_score_lint_status_fails(self):
        steps = [
            VerificationStep(name="lint_ruff", command=[], cwd="", exit_code=1, duration_ms=1, success=False),
        ]
        assert score_lint_status(steps) == 0.0

    def test_score_lint_status_no_lint_steps(self):
        assert score_lint_status([]) == 1.0

    def test_score_type_status_all_pass(self):
        steps = [
            VerificationStep(name="typecheck_mypy", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_type_status(steps) == 1.0

    def test_score_type_status_no_steps(self):
        assert score_type_status([]) == 1.0

    def test_score_build_status_all_pass(self):
        steps = [
            VerificationStep(name="build", command=[], cwd="", exit_code=0, duration_ms=1, success=True),
        ]
        assert score_build_status(steps) == 1.0

    def test_score_build_status_no_steps(self):
        assert score_build_status([]) == 1.0

    def test_score_patch_size_small(self):
        assert score_patch_size("line1\nline2\n") == 1.0

    def test_score_patch_size_giant(self):
        big = "\n".join(f"line{i}" for i in range(300))
        assert score_patch_size(big) < 1.0

    def test_score_patch_size_none(self):
        assert score_patch_size(None) == 1.0

    def test_score_runtime_cost_under_max(self):
        assert score_runtime_cost(10_000) == 1.0

    def test_score_runtime_cost_over_max(self):
        assert score_runtime_cost(200_000) < 1.0

    def test_score_runtime_cost_zero(self):
        assert score_runtime_cost(0) == 1.0

    def test_score_novelty_bonus_all_new(self):
        assert score_novelty_bonus(["a.py", "b.py"], {"c.py"}) == 1.0

    def test_score_novelty_bonus_some_new(self):
        assert score_novelty_bonus(["a.py", "b.py"], {"a.py"}) == pytest.approx(0.5)

    def test_score_novelty_bonus_all_known(self):
        assert score_novelty_bonus(["a.py"], {"a.py"}) == 0.0

    def test_score_novelty_bonus_no_archive(self):
        assert score_novelty_bonus(["a.py", "b.py"]) == 1.0


# --------------- injectable runner ------------


class _RecordingRunner:
    """CommandRunner that records every call and returns programmable results."""

    def __init__(self, results=None, default=CommandResult(returncode=0, stdout=b"", stderr=b"")):
        self._results = list(results or [])
        self._default = default
        self.calls: list[dict] = []

    def run(self, command, *, cwd, timeout, env=None):
        self.calls.append(
            {"command": command, "cwd": cwd, "timeout": timeout, "env": env}
        )
        if self._results:
            return self._results.pop(0)
        return self._default


class TestInjectableRunner:
    def test_injected_runner_receives_command_cwd_timeout_env(self):
        runner = _RecordingRunner(
            results=[CommandResult(returncode=0, stdout=b"ok", stderr=b"")]
        )
        verifier = PythonRepoVerifier(
            steps=[
                {
                    "name": "pytest",
                    "command": ["python", "-m", "pytest"],
                    "expected_exit_code": 0,
                    "timeout": 30,
                }
            ],
            runner=runner,
        )
        verifier.run("/tmp/repo", extra_env={"FOO": "bar"})
        # First call is the pytest step.
        first = runner.calls[0]
        assert first["command"] == ["python", "-m", "pytest"]
        assert first["cwd"] == "/tmp/repo"
        assert first["timeout"] == 30
        assert first["env"]["FOO"] == "bar"
        # extra_env merges over os.environ — typical env vars survive.
        # We don't assert on a specific env var (depends on host), only on the merge.
        assert "FOO" in first["env"]

    def test_no_subprocess_patching_required_for_injected_runner(self):
        """Calling the verifier with an injected runner must not require
        patching ``subprocess.run`` at module scope."""
        runner = _RecordingRunner()
        verifier = PythonRepoVerifier(
            steps=[
                {
                    "name": "lint",
                    "command": ["ruff", "check", "."],
                    "expected_exit_code": 0,
                    "timeout": 10,
                }
            ],
            runner=runner,
        )
        # No `with patch(...)` block.
        result = verifier.run("/tmp/repo")
        assert result.success is True
        assert len(runner.calls) >= 1

    def test_task_id_appears_in_verification_result(self):
        runner = _RecordingRunner()
        verifier = PythonRepoVerifier(
            steps=[{"name": "noop", "command": ["echo", "hi"], "timeout": 5}],
            runner=runner,
        )
        result = verifier.run("/tmp/repo", task_id="task-42")
        assert result.task_id == "task-42"

    def test_task_id_default_is_empty_string(self):
        runner = _RecordingRunner()
        verifier = PythonRepoVerifier(
            steps=[{"name": "noop", "command": ["echo", "hi"], "timeout": 5}],
            runner=runner,
        )
        result = verifier.run("/tmp/repo")
        assert result.task_id == ""

    def test_timeout_via_injected_runner(self):
        """Runner that raises ``subprocess.TimeoutExpired`` is reflected as a
        timeout failure in the step's stderr_excerpt and the step.success."""

        class TimingOutRunner:
            def run(self, command, *, cwd, timeout, env=None):
                raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

        verifier = PythonRepoVerifier(
            steps=[{"name": "slow", "command": ["sleep", "100"], "timeout": 1}],
            runner=TimingOutRunner(),
        )
        result = verifier.run("/tmp/repo")
        assert result.success is False
        assert any("timeout" in f for f in result.failures)
        assert result.steps[0].stderr_excerpt == "timeout"
        assert result.steps[0].success is False


# --------------- changed_files extraction ------------


class _GitOnlyRunner:
    """Runner that returns a programmable result only for ``git`` commands.

    Useful to exercise ``extract_changed_files`` without a real git repo.
    """

    def __init__(self, git_result: CommandResult, raise_on_git: bool = False):
        self.git_result = git_result
        self.raise_on_git = raise_on_git
        self.calls: list[list[str]] = []

    def run(self, command, *, cwd, timeout, env=None):
        self.calls.append(command)
        if command and command[0] == "git":
            if self.raise_on_git:
                raise OSError("git not found")
            return self.git_result
        return CommandResult(returncode=0)


class TestChangedFilesExtraction:
    def test_returns_empty_when_path_is_not_a_git_repo(self, tmp_path):
        # tmp_path has no .git/ — should short-circuit and return [].
        runner = _RecordingRunner()
        files = extract_changed_files(str(tmp_path), runner)
        assert files == []
        # And we didn't even call git, since the .git check failed first.
        assert runner.calls == []

    def test_parses_porcelain_output(self, tmp_path):
        # Make tmp_path look like a git repo to pass the .git check.
        (tmp_path / ".git").mkdir()
        runner = _GitOnlyRunner(
            git_result=CommandResult(
                returncode=0,
                stdout=(
                    b" M src/auth.py\n"
                    b"?? src/new_module.py\n"
                    b"A  src/added.py\n"
                ),
            )
        )
        files = extract_changed_files(str(tmp_path), runner)
        assert files == ["src/auth.py", "src/new_module.py", "src/added.py"]

    def test_parses_rename(self, tmp_path):
        (tmp_path / ".git").mkdir()
        runner = _GitOnlyRunner(
            git_result=CommandResult(
                returncode=0,
                stdout=b"R  src/old.py -> src/new.py\n",
            )
        )
        files = extract_changed_files(str(tmp_path), runner)
        assert files == ["src/new.py"]

    def test_returns_empty_when_git_fails(self, tmp_path):
        (tmp_path / ".git").mkdir()
        runner = _GitOnlyRunner(
            git_result=CommandResult(returncode=128, stdout=b"", stderr=b"fatal: ...")
        )
        files = extract_changed_files(str(tmp_path), runner)
        assert files == []

    def test_returns_empty_when_runner_raises(self, tmp_path):
        (tmp_path / ".git").mkdir()
        runner = _GitOnlyRunner(
            git_result=CommandResult(returncode=0), raise_on_git=True
        )
        files = extract_changed_files(str(tmp_path), runner)
        assert files == []

    def test_changed_files_populated_in_verifier_result(self, tmp_path):
        """Full verifier flow: changed_files comes from git porcelain via the
        injected runner."""
        (tmp_path / ".git").mkdir()

        class StepAndGitRunner:
            def __init__(self):
                self.calls = []

            def run(self, command, *, cwd, timeout, env=None):
                self.calls.append(command)
                if command and command[0] == "git":
                    return CommandResult(
                        returncode=0,
                        stdout=b" M src/handler.py\n M src/router.py\n",
                    )
                # The pytest step succeeds.
                return CommandResult(returncode=0, stdout=b"ok\n", stderr=b"")

        runner = StepAndGitRunner()
        verifier = PythonRepoVerifier(
            steps=[
                {
                    "name": "pytest",
                    "command": ["python", "-m", "pytest"],
                    "timeout": 30,
                }
            ],
            runner=runner,
        )
        result = verifier.run(str(tmp_path), task_id="task-7")
        assert result.task_id == "task-7"
        assert result.changed_files == ["src/handler.py", "src/router.py"]
        # Steps still report the pytest step normally.
        assert len(result.steps) == 1
        assert result.steps[0].success is True

    def test_changed_files_failure_does_not_mask_step_success(self, tmp_path):
        """Even if changed-file extraction errors, the verifier returns the
        real per-step outcome rather than a fake failure."""
        (tmp_path / ".git").mkdir()

        class StepOkGitFailsRunner:
            def __init__(self):
                self.calls = []

            def run(self, command, *, cwd, timeout, env=None):
                self.calls.append(command)
                if command and command[0] == "git":
                    raise RuntimeError("git is broken")
                return CommandResult(returncode=0, stdout=b"ok", stderr=b"")

        runner = StepOkGitFailsRunner()
        verifier = PythonRepoVerifier(
            steps=[{"name": "pytest", "command": ["pytest"], "timeout": 5}],
            runner=runner,
        )
        result = verifier.run(str(tmp_path))
        assert result.success is True
        assert result.changed_files == []


# --------------- SubprocessRunner ------------


class TestSubprocessRunner:
    def test_default_runner_calls_subprocess_run(self):
        runner = SubprocessRunner()
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_result = MagicMock(returncode=0, stdout=b"hi\n", stderr=b"")
            mock_run.return_value = mock_result
            result = runner.run(["echo", "hi"], cwd="/tmp", timeout=5, env={"X": "1"})
            assert result.returncode == 0
            assert result.stdout == b"hi\n"
            kwargs = mock_run.call_args.kwargs
            assert kwargs["shell"] is False
            assert kwargs["cwd"] == "/tmp"
            assert kwargs["timeout"] == 5
            assert kwargs["env"] == {"X": "1"}
            assert kwargs["capture_output"] is True

    def test_default_runner_propagates_timeout_exception(self):
        runner = SubprocessRunner()
        with patch("cl_layer.verify.python_repo.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["x"], timeout=1)
            with pytest.raises(subprocess.TimeoutExpired):
                runner.run(["x"], cwd="/tmp", timeout=1)
