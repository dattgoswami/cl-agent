from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cl_layer.episode.recorder import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.aider import (  # noqa: E402
    AiderChatMessage,
    AiderProcessResult,
    AiderRunner,
    ContextBuilder,
    GitSnapshot,
    load_chat_history_lines,
    map_aider_run,
    parse_git_status_paths,
)
from adapters.aider.log_loader import _flush_buffers  # noqa: E402


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "cl-test@example.com")
    _git(repo, "config", "user.name", "CL Test")
    (repo / "app.py").write_text("print('old')\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "init")


def test_aider_context_builder_integrated_injects_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "PROGRAM.md").write_text("Prefer focused pytest runs.", encoding="utf-8")
    (artifacts / "SKILLS.md").write_text("Use pathlib for paths.", encoding="utf-8")

    builder = ContextBuilder(artifacts)
    baseline = builder.build("Fix the test.", mode="baseline", cwd="/repo")
    integrated = builder.build("Fix the test.", mode="integrated", cwd="/repo")

    assert baseline.message == "Fix the test."
    assert "PROGRAM.md" in integrated.message
    assert "SKILLS.md" in integrated.message
    assert "## Current Task" in integrated.message
    assert integrated.cli_args() == ["--message", integrated.message]


def test_aider_chat_history_parser_keeps_coarse_roles() -> None:
    messages = load_chat_history_lines(
        [
            "# aider chat started at 2026-04-20 10:00:00\n",
            "> Aider v0.47.2\n",
            "> Git repo: .git with 2 files\n",
            "\n",
            "#### Fix app.py\n",
            "\n",
            "I updated the file.\n",
            "\n",
            "> Running pytest -q\n",
            "> 1 passed\n",
            "> exit code: 0\n",
        ]
    )

    assert [message.role for message in messages] == ["tool", "user", "assistant", "tool"]
    assert messages[1].content == "Fix app.py"
    assert "I updated" in messages[2].content
    assert "pytest -q" in messages[3].content


def test_aider_chat_history_flushes_invariant_violation_without_crashing(caplog) -> None:
    messages: list[AiderChatMessage] = []

    _flush_buffers(
        messages,
        assistant=["assistant text\n"],
        user=["user text\n"],
        tool=["> tool text\n"],
    )

    assert [message.role for message in messages] == ["assistant", "user", "tool"]
    assert "parser invariant violated" in caplog.text


def test_parse_git_status_paths_handles_renames_and_untracked_files() -> None:
    status = " M src/app.py\n?? docs/new.md\nR  old.py -> new.py\n"
    assert parse_git_status_paths(status) == ["docs/new.md", "new.py", "src/app.py"]


def test_map_aider_run_records_command_diff_messages_and_tests() -> None:
    started = datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 4, 20, 10, 1, tzinfo=timezone.utc)
    diff = "--- a/app.py\n+++ b/app.py\n@@\n-print('old')\n+print('new')\n"
    chat = [
        AiderChatMessage(role="user", content="Fix app.py", index=0),
        AiderChatMessage(role="assistant", content="I updated app.py.", index=1),
        AiderChatMessage(role="tool", content="Running pytest -q\n1 passed\nexit code: 0", index=2),
    ]

    mapped = map_aider_run(
        args=["aider", "--message", "Fix app.py"],
        returncode=0,
        stdout="Tokens: 100 sent, 20 received.",
        stderr="",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
        chat_messages=chat,
        changed_files=["app.py"],
        patch_text=diff,
    )

    assert mapped.outcome.status == "completed"
    assert mapped.outcome.tests_passed is True
    assert mapped.outcome.files_touched == ["app.py"]
    assert mapped.outcome.final_response == "I updated app.py."
    assert mapped.patch_hash and mapped.patch_hash.startswith("sha256:")
    assert mapped.test_trace == [
        {"command": "pytest -q", "exit_code": 0, "source": "aider_chat_history"}
    ]
    assert [event.kind for event in mapped.events].count("command_execution") == 2
    file_change = next(event for event in mapped.events if event.kind == "file_change")
    assert file_change.payload["patch_hash"] == mapped.patch_hash


def test_map_aider_run_captures_adjacent_running_lines() -> None:
    started = datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 4, 20, 10, 1, tzinfo=timezone.utc)

    mapped = map_aider_run(
        args=["aider", "--message", "Run checks"],
        returncode=0,
        stdout="",
        stderr="",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
        chat_messages=[
            AiderChatMessage(
                role="tool",
                content="Running: ruff check\nexit code: 0\nRunning pytest -q\nexit code: 1",
                index=0,
            )
        ],
    )

    commands = [
        event.payload["command"]
        for event in mapped.events
        if event.kind == "command_execution" and event.payload.get("command")
    ]
    exit_codes = [
        event.payload.get("exit_code")
        for event in mapped.events
        if event.kind == "command_execution" and event.payload.get("command")
    ]
    assert commands == ["ruff check", "pytest -q"]
    assert exit_codes == [0, 1]
    assert mapped.outcome.tests_passed is False
    assert mapped.outcome.status == "failed"


def test_map_aider_run_derives_failed_and_partial_outcomes() -> None:
    started = datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 4, 20, 10, 1, tzinfo=timezone.utc)

    failed = map_aider_run(
        args=["aider", "--message", "Fix app.py"],
        returncode=2,
        stdout="",
        stderr="model error",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
    )
    partial = map_aider_run(
        args=["aider", "--message", "Fix app.py"],
        returncode=2,
        stdout="",
        stderr="model error",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
        changed_files=["app.py"],
    )

    assert failed.outcome.status == "failed"
    assert failed.outcome.escalation_reason == "aider exited with code 2"
    assert partial.outcome.status == "partial"
    assert partial.outcome.files_touched == ["app.py"]


def test_map_aider_run_stderr_failure_without_final_response_is_failed() -> None:
    started = datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 4, 20, 10, 1, tzinfo=timezone.utc)

    mapped = map_aider_run(
        args=["aider", "--message", "Fix app.py"],
        returncode=0,
        stdout="",
        stderr="Error: rate limit exceeded",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
    )

    assert mapped.outcome.status == "failed"
    assert mapped.outcome.escalation_reason == "stderr indicates failure"


def test_map_aider_run_failure_text_and_failed_tests_without_edits_are_failed() -> None:
    started = datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 4, 20, 10, 1, tzinfo=timezone.utc)

    final_failure = map_aider_run(
        args=["aider", "--message", "Fix app.py"],
        returncode=0,
        stdout="",
        stderr="",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
        chat_messages=[
            AiderChatMessage(role="assistant", content="I cannot complete this task.", index=0)
        ],
    )
    failed_test = map_aider_run(
        args=["aider", "--message", "Fix app.py"],
        returncode=0,
        stdout="",
        stderr="",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
        chat_messages=[
            AiderChatMessage(
                role="tool",
                content="Running pytest -q\n1 failed\nexit code: 1",
                index=0,
            )
        ],
    )
    partial_failed_test = map_aider_run(
        args=["aider", "--message", "Fix app.py"],
        returncode=0,
        stdout="",
        stderr="",
        cwd="/repo",
        started_at=started,
        ended_at=ended,
        chat_messages=[
            AiderChatMessage(
                role="tool",
                content="Running pytest -q\n1 failed\nexit code: 1",
                index=0,
            )
        ],
        changed_files=["app.py"],
    )

    assert final_failure.outcome.status == "failed"
    assert failed_test.outcome.status == "failed"
    assert failed_test.outcome.tests_passed is False
    assert partial_failed_test.outcome.status == "partial"


def test_aider_runner_does_not_attribute_preexisting_untracked_files(tmp_path: Path) -> None:
    runner = AiderRunner(tmp_path / "episodes.jsonl")
    before = GitSnapshot(
        is_repo=True,
        head="abc123",
        status_text="?? notes.txt\n",
        dirty_files=["notes.txt"],
        diff_text=None,
    )
    after = GitSnapshot(
        is_repo=True,
        head="abc123",
        status_text="?? notes.txt\n",
        dirty_files=["notes.txt"],
        diff_text=None,
    )

    patch_text, changed_files = runner._derive_patch_and_files(before, after, cwd=None)

    assert patch_text is None
    assert changed_files == []


def test_aider_runner_roundtrip_ignores_preexisting_untracked_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "notes.txt").write_text("pre-existing scratch\n", encoding="utf-8")

    def fake_runner(args, cwd, env):
        chat_path = Path(args[args.index("--chat-history-file") + 1])
        chat_path.write_text("#### Inspect only\n\nI did not change files.\n", encoding="utf-8")
        return AiderProcessResult(args=list(args), returncode=0, stdout="", stderr="")

    runner = AiderRunner(
        tmp_path / "episodes.jsonl",
        capture_dir=tmp_path / "captures",
        command_runner=fake_runner,
    )
    episode = runner.run(
        "Inspect only",
        task_id="preexisting-untracked",
        task_domain="python",
        cwd=str(repo),
        mode="baseline",
    )

    assert episode.outcome.status == "completed"
    assert episode.outcome.files_touched == []
    assert episode.patch_text is None


def test_aider_runner_uses_fake_subprocess_git_diff_and_recorder_roundtrip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    episodes_path = tmp_path / "episodes.jsonl"

    def fake_runner(args, cwd, env):
        assert cwd == str(repo)
        assert "--no-auto-commits" in args
        assert "--no-dirty-commits" in args
        assert "--no-restore-chat-history" in args
        assert "--no-gitignore" in args
        assert "--message" in args
        assert args[args.index("--message") + 1] == "Update app.py"

        chat_path = Path(args[args.index("--chat-history-file") + 1])
        chat_path.write_text(
            "\n".join(
                [
                    "# aider chat started at 2026-04-20 10:00:00",
                    "#### Update app.py",
                    "",
                    "I updated app.py.",
                    "",
                    "> Running pytest -q",
                    "> 1 passed",
                    "> exit code: 0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        Path(cwd, "app.py").write_text("print('new')\n", encoding="utf-8")
        return AiderProcessResult(args=list(args), returncode=0, stdout="done\n", stderr="")

    runner = AiderRunner(
        episodes_path,
        capture_dir=tmp_path / "captures",
        command_runner=fake_runner,
    )
    result = runner.run_with_result(
        "Update app.py",
        task_id="task-aider",
        task_domain="python",
        cwd=str(repo),
        mode="baseline",
        task_description="",
    )

    episode = result.episode
    assert episode.agent_surface == "aider"
    assert episode.run_id == Path(result.chat_history_file).parent.name
    assert episode.task_description == ""
    assert episode.mode == "baseline"
    assert episode.outcome.status == "completed"
    assert episode.outcome.tests_passed is True
    assert episode.outcome.files_touched == ["app.py"]
    assert episode.patch_text and "+print('new')" in episode.patch_text
    assert episode.patch_hash and episode.patch_hash.startswith("sha256:")

    loaded = EpisodeRecorder(episodes_path).load_all()
    assert len(loaded) == 1
    assert loaded[0].episode_id == episode.episode_id
    assert loaded[0].patch_text == episode.patch_text


def test_aider_runner_integrated_message_includes_cl_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "PROGRAM.md").write_text("Remember prior repo lesson.", encoding="utf-8")

    seen_message: dict[str, str] = {}

    def fake_runner(args, cwd, env):
        message = args[args.index("--message") + 1]
        seen_message["message"] = message
        chat_path = Path(args[args.index("--chat-history-file") + 1])
        chat_path.write_text("#### Do it\n\nDone.\n", encoding="utf-8")
        return AiderProcessResult(args=list(args), returncode=0, stdout="", stderr="")

    runner = AiderRunner(
        tmp_path / "episodes.jsonl",
        artifacts_dir=artifacts,
        capture_dir=tmp_path / "captures",
        command_runner=fake_runner,
    )
    episode = runner.run(
        "Do it",
        task_id="integrated-aider",
        task_domain="python",
        cwd=str(repo),
        mode="integrated",
    )

    assert "PROGRAM.md" in seen_message["message"]
    assert "Remember prior repo lesson." in seen_message["message"]
    assert "## Current Task" in seen_message["message"]
    assert episode.outcome.status == "completed"
