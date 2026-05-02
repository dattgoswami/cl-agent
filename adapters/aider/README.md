# Aider Adapter

Conservative Aider adapter that records one-shot Aider CLI runs as normalized CL episodes.

## Evidence Source

The first implementation does not depend on Aider internals or a Python event API. It captures:

| Evidence | Source |
|----------|--------|
| Aider invocation, exit code, stdout, stderr | injected subprocess runner |
| Chat/user/tool transcript | run-specific `--chat-history-file` |
| Changed files and patch | `git status --porcelain`, `git diff --binary` before/after |
| Optional test/lint evidence | exact Aider tool-output lines like `Running pytest -q` with an exit-code marker |

This keeps attribution coarse: code changes come from git, not from parsing assistant prose.

## Files

| File | Purpose |
|------|---------|
| `context_builder.py` | Builds baseline/integrated one-shot Aider prompts |
| `log_loader.py` | Parses `.aider.chat.history.md`-style markdown into coarse messages |
| `item_mapper.py` | Maps subprocess, chat, and git evidence into CL events/outcome |
| `runner.py` | Injectable CLI runner with git snapshots and recorder append |

## Setup

```bash
pip install -e path/to/cl-agent/cl-layer
pip install aider-chat  # only needed for live runs, not tests
```

Tests use fake subprocess results and temporary git repos. They do not require Aider, model keys, package installs, or network.

## Usage

```python
from adapters.aider import AiderRunner

runner = AiderRunner(
    "data/episodes.jsonl",
    artifacts_dir="data",
)

episode = runner.run(
    "Fix the failing pytest case.",
    task_id="task-001",
    task_domain="python",
    mode="baseline",
    cwd="/path/to/git/repo",
    model="sonnet",
    test_cmd="pytest -q",
    auto_test=True,
)
```

## Modes

| Mode | Behavior |
|------|----------|
| `baseline` | Sends only the task prompt. The runner passes `--no-restore-chat-history` and uses run-specific history files. |
| `integrated` | Prepends `PROGRAM.md` and `SKILLS.md` from `artifacts_dir` to the one-shot Aider message, then sends the current task. |

Both modes avoid Aider history carryover by default. Integrated mode gets CL context explicitly through the supported `--message` prompt path, not via `.aider.conf.yml` or user-level config.

## Safety Defaults

The runner adds these Aider flags by default:

```text
--no-auto-commits
--no-dirty-commits
--no-restore-chat-history
--no-gitignore
--no-analytics
--no-check-update
--no-show-release-notes
--yes-always
--no-pretty
--no-stream
```

Aider normally enables auto-commits and dirty pre-edit commits. The adapter disables both so the post-run working tree diff remains visible for CL capture and so tests do not create commits. If you override this, the runner can still capture a commit-to-commit diff when `HEAD` changes, but the safer default is uncommitted patch capture.

`--no-gitignore` avoids modifying `.gitignore` just to add `.aider*`. The adapter writes chat, input, and LLM history into `aider-captures/<run_id>/` next to the episode log unless a custom `capture_dir` is provided.

## Mapping

| Aider evidence | CL mapping |
|----------------|------------|
| CLI subprocess | `command_execution` with args, cwd, exit code, stdout/stderr excerpts |
| Aider chat history user/assistant/tool chunks | `agent_message` |
| Aider tool output line `Running <cmd>` | additional `command_execution` |
| test/lint command with parsed exit code | `test_trace`, `outcome.tests_passed`, verification summary |
| clean-worktree post-run git diff | `file_change`, `patch_text`, `patch_hash` |
| changed paths from git status | `outcome.files_touched` |

If the worktree was dirty before the run, the adapter records only newly dirty paths and skips `patch_text` to avoid attributing pre-existing changes to Aider.

## Limitations

- No native Aider Python event stream is used.
- Assistant messages are preserved as coarse chat evidence only.
- Test/lint commands are detected only from explicit Aider tool-output lines, not arbitrary prose.
- Untracked file contents may appear as changed paths without patch text unless git can expose them in the final diff.
- Long or complex integrated context may eventually need `--message-file`; the first implementation uses `--message` for simplicity and testability.

## Tests

```bash
cd cl-layer
python -m pytest tests/test_aider_adapter.py -v
python -m pytest tests/test_aider_adapter.py tests/test_pi_mono_adapter.py tests/test_hermes_agent_adapter.py -q
```
