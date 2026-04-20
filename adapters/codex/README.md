# Codex Adapter

Thin adapter that drives the Codex SDK / app-server and maps
structured `ThreadItem` outputs into normalized CL-layer episodes.

## Files

| File | Purpose |
|------|---------|
| `sdk_runner.py` | `CodexRunner` — owns the full run lifecycle |
| `item_mapper.py` | Translates `ThreadItem` list → `(events, outcome)` |
| `context_builder.py` | Builds `RunContext` from task + substrate artifacts |

## Setup

```bash
# Install the CL substrate
pip install -e path/to/continual-learning-agent/cl-layer

# Install the Codex Python SDK
pip install -e path/to/continual-learning-agent/codex/sdk/python

# The Codex app-server must be running locally.
# See codex/codex-rs/README.md for how to start it.
```

## Quick start

```python
from adapters.codex.sdk_runner import CodexRunner

runner = CodexRunner(
    episodes_path="data/episodes.jsonl",
    artifacts_dir="data/",          # reads PROGRAM.md / SKILLS.md from here
    default_model="o4-mini",
)

episode = runner.run(
    task_prompt="Add a /healthz endpoint to main.py",
    task_id="task-001",
    task_domain="fastapi",
    mode="baseline",                 # or "integrated"
    cwd="/path/to/working/repo",
)

print(episode.outcome.status)
print(episode.outcome.files_touched)
```

After a run, inspect the artifacts:

```
data/
├── episodes.jsonl   # append-only episode log
├── SKILLS.md        # distilled patterns (written by distill/skills.py)
├── DREAMS.md        # session summaries (written by distill/dreams.py)
└── PROGRAM.md       # next-run context (written by distill/program.py)
```

## Operating modes

### `baseline`

- Thread is `ephemeral=True` — Codex native memory is minimized.
- `PROGRAM.md` and `SKILLS.md` are **not** injected as developer instructions.
- The substrate is the only deliberate cross-session learning signal.
- Use for controlled experiments.

### `integrated`

- Thread is **not** ephemeral — Codex native memory persists.
- If `PROGRAM.md` and `SKILLS.md` exist under `artifacts_dir`, they are
  injected as developer instructions before the task runs.
- Use for practical daily usage where both memory systems coexist.

## Item types mapped

| Codex `type` | Episode `kind` |
|--------------|----------------|
| `commandExecution` | `command_execution` |
| `fileChange` | `file_change` |
| `mcpToolCall` | `mcp_tool_call` |
| `agentMessage` | `agent_message` |
| all others | skipped |

## What is not yet built

- Distillation write-back after a run (call `distill/skills.py`,
  `distill/dreams.py`, `distill/program.py` manually or from a wrapper).
- Reward derivation (see `cl_layer/eval/` — Phase 4).
- MCP interface (Phase 6).
