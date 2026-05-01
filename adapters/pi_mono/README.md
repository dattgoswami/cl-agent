# Pi Monorepo Adapter

Import-first adapter that converts Pi coding-agent session JSONL files into normalized CL episodes, with an optional thin CLI runner for integrated mode.

## Files

| File | Purpose |
|------|---------|
| `session_loader.py` | Parses Pi session JSONL; `import_session_jsonl`, `append_session_episode`, `session_to_episode` |
| `item_mapper.py` | Maps Pi message entries → `EpisodeEvent` list and derives `EpisodeOutcome` |
| `context_builder.py` | Builds `PiRunContext`; injects `PROGRAM.md`/`SKILLS.md` as `--append-system-prompt` in integrated mode |
| `runner.py` | `PiCliRunner` — thin, injectable CLI wrapper; does not parse live output into an `Episode` |
| `time_utils.py` | Shared `parse_pi_datetime`; handles ISO strings and Unix-ms integers uniformly |

## Setup

```bash
# Install the CL substrate (required before any import)
pip install -e path/to/cl-agent/cl-layer

# No Pi TypeScript build step is required for any Python import or test.
# The adapter reads Pi's session JSONL files directly.
```

## Locating the Pi session file

Pi writes an append-only JSONL file for each session. The exact path depends on your Pi version and configuration — consult `pi-mono/packages/coding-agent/docs/session.md` or run:

```bash
pi sessions list          # if your Pi build exposes this command
ls ~/.pi/agent/sessions/  # approximate default; may differ by install
```

The path shown in code examples below is approximate. Substitute the real path from the above.

## Import Usage

### One-shot import

```python
from adapters.pi_mono import import_session_jsonl

episode = import_session_jsonl(
    "/path/to/pi/session.jsonl",   # see "Locating the Pi session file" above
    task_id="task-001",
    task_domain="python",
    task_description="Fix the failing pytest case",
    mode="baseline",
)

print(episode.outcome.status)         # "completed" | "partial" | "failed"
print(episode.outcome.files_touched)  # only proven successful mutations (edit/write)
print(episode.base_model_id)          # e.g. "anthropic/claude-sonnet-4-5"
```

### Import and append to episodes.jsonl

```python
from adapters.pi_mono import append_session_episode

episode = append_session_episode(
    "/path/to/pi/session.jsonl",
    "data/episodes.jsonl",
    task_id="task-001",
    task_domain="python",
    task_description="Fix the failing pytest case",
)
```

### Selecting a specific branch

Pi v2/v3 session files are append-only trees. By default the importer walks from the last entry back to the root, mirroring Pi's active-branch behavior. Pass `branch_leaf_id` to import a different branch:

```python
episode = import_session_jsonl(
    "/path/to/pi/session.jsonl",
    task_id="task-001",
    task_domain="python",
    task_description="Fix the failing pytest case",
    branch_leaf_id="entry-id-of-leaf",  # from session JSONL entry "id" field
)
```

Legacy linear sessions (entries with no `id` field) are returned in file order and do not support `branch_leaf_id`.

## End-to-End Loop

The full CL loop for Pi mirrors the Codex loop except that capture is post-hoc (import from JSONL) rather than live:

```
1. Run Pi on a task (Pi writes session JSONL automatically).

2. Import the session:
       from adapters.pi_mono import append_session_episode
       append_session_episode("session.jsonl", "data/episodes.jsonl", ...)

3. Distil knowledge artifacts from accumulated episodes:
       from cl_layer.episode.recorder import EpisodeRecorder
       from cl_layer.distill.skills import distill_skills, distill_warnings, render_skills_md
       from cl_layer.distill.dreams import summarize_session, render_dreams_md
       from cl_layer.distill.program import render_program_md
       from pathlib import Path

       episodes = EpisodeRecorder("data/episodes.jsonl").load_all()
       skills   = distill_skills(episodes)
       warnings = distill_warnings(episodes)
       summary  = summarize_session(episodes)

       Path("data/SKILLS.md").write_text(render_skills_md(skills))
       Path("data/DREAMS.md").write_text(render_dreams_md(summary))
       Path("data/PROGRAM.md").write_text(render_program_md("python", skills, warnings))

4. On the next Pi run, pass the ContextBuilder output to inject prior knowledge:
       from adapters.pi_mono import ContextBuilder, PiCliRunner

       runner = PiCliRunner(artifacts_dir="data/")
       result = runner.run(
           "Add a /readyz endpoint to main.py.",
           mode="integrated",          # injects PROGRAM.md + SKILLS.md via --append-system-prompt
           cwd="/path/to/repo",
       )
       # Then import the new session JSONL and repeat from step 2.
```

After step 3 the artifacts directory should contain:

```
data/
├── episodes.jsonl   # append-only episode log
├── SKILLS.md        # distilled reusable patterns
├── DREAMS.md        # session summaries: what worked, what failed
└── PROGRAM.md       # injected context for the next run
```

## Event Mapping

| Pi evidence | Episode event |
|-------------|---------------|
| `message.role == "bashExecution"` | `command_execution` |
| `toolResult` for `toolName == "bash"` paired with a `bash` tool call | `command_execution` |
| `toolResult` for `read`, `write`, or `edit` with a proven path from the tool call | `file_change` |
| `edit` result `details.diff` | `file_change.payload.diff`, plus episode `patch_text` / `patch_hash` |
| user, assistant, custom, branch summary, and compaction summary messages | `agent_message` |

Ordinary Pi tools are not mapped as `mcp_tool_call`. Pi's inspected coding-agent surface does not expose MCP events in these session files.

`EpisodeOutcome.files_touched` records proven successful mutations only (`edit` and `write` with `isError=False`). Read-only evidence remains available as `file_change` events with `payload.mutating == False`.

## Outcome Derivation

The importer derives outcome conservatively:

- assistant `stopReason` of `error` or `aborted` → `failed`
- cancelled `bashExecution` → `failed`
- nonzero bash exit codes or bash tool errors → `partial` (unless a stronger explicit failure exists)
- test-like commands (`pytest`, `npm test`, `npm run check`, `go test`, `cargo test`) → `tests_passed` set from the latest such command
- final assistant text containing failure language → `partial`

## Context Injection

`ContextBuilder` supports two modes:

- `baseline`: no CL artifacts are injected. `PiCliRunner` adds `--no-session` so live runs are ephemeral (assumed from `src/cli/args.ts` inspection; verify against your Pi build if the flag is not recognized).
- `integrated`: `PROGRAM.md` and `SKILLS.md` from `artifacts_dir` are appended to Pi's system prompt via `--append-system-prompt` (grounded in `src/core/system-prompt.ts`, `src/core/resource-loader.ts`, and `src/cli/args.ts`). The adapter does not write `.pi/APPEND_SYSTEM.md`, user settings, skills, or extensions.

```python
from adapters.pi_mono import ContextBuilder

builder = ContextBuilder(artifacts_dir="data/")
ctx = builder.build("Fix the flaky test.", mode="integrated", cwd="/path/to/repo")

print(ctx.cli_args())
# ['--append-system-prompt', '## Prior Context (PROGRAM.md)\n\n...']
```

## Live Runner Status

`PiCliRunner` builds and dispatches a Pi CLI command through an injectable `CommandRunner`. It is suitable for integration wrappers and dry-run tests, but **capture is always session-import based** — `PiCliRunner` does not parse live Pi output into an `Episode`. After `runner.run(...)` completes, call `import_session_jsonl` on the JSONL Pi wrote.

Direct Pi RPC / SDK driving is not yet implemented.

## Source Surface

This adapter is grounded in Pi's `packages/coding-agent` surface:

- `docs/session.md` — session headers, v3 message entries, branch trees, compaction entries
- `src/core/session-manager.ts` — append-only JSONL with `id`/`parentId` trees; active-leaf reconstruction
- `src/core/messages.ts` — coding-agent roles: `bashExecution`, `custom`, `branchSummary`, `compactionSummary`
- `src/core/system-prompt.ts`, `src/core/resource-loader.ts`, `src/cli/args.ts` — `--append-system-prompt` injection path

No Pi TypeScript code is imported or built by the Python tests.

## Running the Tests

```bash
cd cl-layer
python -m pytest tests/test_pi_mono_adapter.py -v
# 11 tests — no network, no Pi binary, no API keys, no real model calls
```

To run the full suite:

```bash
cd cl-layer
python -m pytest tests/ -q
```
