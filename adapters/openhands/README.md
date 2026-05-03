# OpenHands Adapter

Import OpenHands V1 conversation exports into normalized CL episodes.

## Capture boundary

The adapter targets the **V1 exported conversation format** — zip files or
filesystem directories written by `FilesystemEventService`:

```
# ZIP export (from API or manual export)
conversation.zip
├── meta.json          # conversation metadata
├── event_000001_<id>.json
├── event_000002_<id>.json
└── ...

# Filesystem persistence (live runs)
{persistence_dir}/{user_id}/v1_conversations/{conversation_id}/
├── meta.json
├── {event_id_hex}.json
└── ...
```

No OpenHands runtime, Docker, app server, or model API keys are required to
import.

Malformed event JSON is treated as a hard import error.  The importer fails
fast rather than silently emitting a partial episode from corrupted training
data.

## Setup

```bash
# Install the cl-layer (from repo root)
pip install -e cl-layer/
```

The adapter has no additional runtime dependencies.  `difflib` (stdlib) is
used to construct file diffs from FileEditorObservation `old_content` /
`new_content` fields.

## Quick start

```python
from adapters.openhands import import_conversation, append_conversation_episode

# Import a single conversation zip
episode = import_conversation(
    "exports/conv-abc123.zip",
    task_id="fix-issue-42",
    task_domain="python",
    mode="baseline",           # or "integrated"
)

# Import and append to a JSONL store
episode = append_conversation_episode(
    "exports/conv-abc123.zip",
    "data/episodes.jsonl",
    task_id="fix-issue-42",
    task_domain="python",
)

# Batch import via OpenHandsImporter
from adapters.openhands import OpenHandsImporter

importer = OpenHandsImporter(
    episodes_path="data/episodes.jsonl",
    artifacts_dir="data/",     # reads PROGRAM.md / SKILLS.md for integrated mode
)
episode = importer.import_conversation(
    "exports/conv-abc123.zip",
    task_id="task-001",
    task_domain="python",
    mode="integrated",
)
```

## V1 event format

Every serialized event is a JSON object with `id`, `timestamp`, and `source`.
The adapter infers the event type by duck-typing on payload keys and V1
top-level `kind` discriminators:

| Key present | Inferred type |
|---|---|
| `action` (nested dict) | ActionEvent — `action.kind` identifies kind |
| `observation` (nested dict) | ObservationEvent — paired with action_id |
| `llm_message` | MessageEvent |
| `system_prompt` | SystemEvent |
| `kind=="ConversationStateUpdateEvent"` | Conversation state, execution status, token stats |
| `kind=="ConversationErrorEvent"` / `kind=="ServerErrorEvent"` | Fatal error signal |
| `kind=="HookExecutionEvent"` | Hook execution telemetry |
| `condensed_events` | CondensationEvent (skipped) |
| `reason` + `source` | PauseEvent (skipped) |
| `source=="agent"` + `error` + `tool_call_id` | AgentErrorEvent |

### Action discriminators (`action.kind`)

| OpenHands action type | CL EventKind |
|---|---|
| `ExecuteBashAction` | `command_execution` |
| `TerminalAction` | `command_execution` |
| `FileEditorAction` (mutating: create / str_replace / insert / undo_edit) | `file_change` |
| `StrReplaceEditorAction` (mutating) | `file_change` |
| `PlanningFileEditorAction` (mutating) | `file_change` |
| `FileEditorAction` (view) | *(skipped — not a mutation)* |
| `MCPToolAction` | `mcp_tool_call` |
| `FinishAction` | outcome.final_response + `agent_message` |
| `Browser*` (all browser action types) | `agent_message` |
| `ThinkAction` | `agent_message` |
| `TaskTrackerAction` / `GlobAction` / `GrepAction` | `agent_message` |
| Any other / future action type | `agent_message` |

### Observation pairing

Observations are paired with their action by `action_id → action.id`.  The
mapper builds an observation index in a first pass, then processes action
events in document order.  Unpaired observations (no matching action in the
event list) are silently skipped to avoid double-counting.

## Mode semantics

### Baseline

- Fresh conversation, isolated persistence directory / user ID.
- No CL artifacts injected.
- OpenHands native memory / microagents may still be active; attribute
  improvements to the substrate only after controlling for native memory.

### Integrated

- CL artifacts (`PROGRAM.md`, `SKILLS.md`) are copied to namespaced files
  (`cl-program.md`, `cl-skills.md`) under `<repo_root>/.openhands/skills/`
  before the run starts.  Existing user-maintained OpenHands skill files named
  `PROGRAM.md` or `SKILLS.md` are preserved.
- OpenHands reads `.openhands/skills/` as a custom microagent instruction
  directory, so the artifacts become part of the agent's context.
- `ContextBuilder.build(mode="integrated", cwd=..., inject_dir=...)` handles
  the file writing; call it before launching the run.

**Attribution caveat**: OpenHands has its own native memory/microagent state.
Compare baseline vs integrated runs with isolated persistence dirs to attribute
improvements correctly.  Do not share persistence dirs between modes.

## Patch text and diff

When `FileEditorObservation` provides `old_content` and `new_content`, the
adapter constructs a unified diff with `difflib.unified_diff` and stores it in
`Episode.patch_text`.  Multiple file diffs are concatenated.
`Episode.patch_hash` is `sha256:<hex>` of `patch_text`.

When observations are absent (no paired observation, or old/new content
missing), no diff is generated.  **Diffs are never fabricated from action
fields alone.**

## Outcome status derivation

| Signal | Status |
|---|---|
| `FinishAction` seen (final_response set) | `completed` |
| `ConversationStateUpdateEvent` execution_status=finished | `completed` |
| Any of the above + prior command failure | `partial` |
| `ConversationStateUpdateEvent` nonterminal execution_status | `partial` |
| Command failures without FinishAction | `partial` |
| AgentErrorEvent + prior successful events | `partial` |
| Events present without any finish/error signal | `partial` |
| No events at all | `failed` |
| Only an AgentErrorEvent | `failed` |

`reward` is always `None` at record time — filled by the evaluator.

## Metadata

- `base_model_id` is read from `meta.json` in this order:
  `llm_model`, `model`, `base_model`, `base_model_id`, then nested
  `llm_config.model` / `llm_config.model_name`.
- `cost_tokens_prompt` and `cost_tokens_completion` are read from V1
  `ConversationStateUpdateEvent` stats snapshots.  These are accumulated
  counters, so repeated stats events replace the previous value rather than
  being summed.  If no stats event is present, `meta.metrics` is used as a
  fallback when available.
- `HookExecutionEvent` payloads preserve V1 hook fields including
  `hook_event_type`, `hook_command`, `success`, `blocked`, `exit_code`,
  `reason`, `tool_name`, `action_id`, `message_id`, `stdout`, `stderr`,
  `error`, `additional_context`, and `hook_input`.

## Known limitations

- **browser screenshots**: `BrowserObservation.screenshot_data` (base64) is
  not stored in the episode payload.  Only the text output is captured.
- **thinking blocks**: `ActionEvent.thinking_blocks` are not stored in episode
  payloads.  The `thought` field (ActionEvent.thought[].text) is captured.
- **live execution deferred**: `OpenHandsLiveRunner` is a documented stub.
  The V1 import path is stable; live execution requires an injectable API
  client and a running app server.  The V1 export download route is
  `GET /api/v1/app-conversations/{conversation_id}/download`; downloaded zip
  bytes should be written to a temporary `.zip` before calling the importer.
- **V0 legacy schema**: V0 `openhands/core/schema/action.py` / `observation.py`
  are not supported.  They are deprecated in the local repo.
- **UserRejectObservation**: Ignored (no CL event emitted).

## Follow-up work

1. Implement `OpenHandsLiveRunner` with injectable API client.
2. Handle UserRejectObservation as `human_feedback` event.
3. Store browser screenshot artifact paths (not base64 content) in payload.
4. Add `benchmark_split` and `task_tags` extraction from meta.json.
