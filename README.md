# cl-agent — Continual-Learning Substrate for Coding Agents

A **Codex-first continual-learning substrate** that attaches to coding agents you already use, records what they tried, distills what worked, and injects that knowledge forward so the agent measurably improves across sessions — without model fine-tuning.

> The durable asset is the learning loop, not the runtime shell.

**Paper:** [cl-agent: A Continual-Learning Substrate for Coding Agents](paper/cl_agent_paper_pub.pdf)

---

## What This Is

This repo ships two things:

| Package | Path | Purpose |
|---------|------|---------|
| **CL Layer** | `cl-layer/` | Core substrate: episode capture, replay buffer, rule-based distillation, evaluation hooks |
| **Codex Adapter** | `adapters/codex/` | Thin adapter that drives Codex via its Python SDK and maps structured `ThreadItem` outputs into normalized episodes |

It is **not** a new coding agent, a replacement for Codex, or a generic orchestration framework. It is the connective tissue between existing agent surfaces and a durable learning loop.

---

## Why This Exists

A survey of 30+ agent, memory, and orchestration repos found the ecosystem strong in pieces but weak in the seams. Seven gaps were consistently unaddressed across all of them:

1. No framework-agnostic continual-learning substrate
2. No shared run or trajectory format
3. No real coding-agent long-term memory (tied to repeated sessions, not just user preferences)
4. No memory interoperability across agent surfaces
5. No safe promotion and rollback of learned behavior
6. No cross-framework eval and replay
7. No portable multi-agent memory-sharing model

This project targets those gaps with a narrow, falsifiable v1 claim:

> A thin substrate built from episode recording, replay, rule-based skill distillation, and context injection can improve coding-agent performance across repeated tasks in a narrow domain without model fine-tuning.

---

## The Four Pillars

Every piece of this project serves exactly one of four responsibilities:

```
1. Episode capture   — record what the agent did, with raw evidence
2. Replay            — surface relevant prior failures and successes
3. Distillation      — extract inspectable, durable knowledge artifacts
4. Evaluation        — measure whether the agent is actually improving
```

Removing any one pillar degrades the project into something less useful: without capture it is just prompting; without replay just logging; without distillation just an event store; without metrics just a memory feature.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Agent Surface                                              │
│                                                            │
│  Codex (v1)   ·   Hermes (planned)   ·   others later     │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│ Thin Adapter   (adapters/codex/)                           │
│                                                            │
│  context_builder.py  →  assembles run context explicitly   │
│  sdk_runner.py       →  drives Codex thread lifecycle      │
│  item_mapper.py      →  maps ThreadItems → episode events  │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│ CL Layer   (cl-layer/)                                     │
│                                                            │
│  episode/   — normalized schema + JSONL recorder           │
│  replay/    — domain-heuristic buffer, no embeddings       │
│  distill/   — rule-based skills, dreams, program writers   │
│  eval/      — operational + research metrics  (Phase 4)    │
│  mcp/       — MCP server interface            (Phase 6)    │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│ Persistent Artifacts                                       │
│                                                            │
│  episodes.jsonl   SKILLS.md   DREAMS.md   PROGRAM.md       │
└────────────────────────────────────────────────────────────┘
```

The core learning artifacts are plain files — human-readable, git-diffable, and consumable by any agent that can read a file. The adapter is the only surface-specific code.

---

## Repository Layout

```
cl-agent/
│
├── cl-layer/                       # installable Python package
│   ├── pyproject.toml
│   └── cl_layer/
│       ├── episode/
│       │   ├── schema.py           # EpisodeEvent, EpisodeOutcome, Episode
│       │   └── recorder.py         # append-only JSONL store
│       ├── replay/
│       │   └── buffer.py           # domain/mode-filtered replay queries
│       └── distill/
│           ├── skills.py           # rule-based skill candidates → SKILLS.md
│           ├── dreams.py           # session summary → DREAMS.md
│           └── program.py          # next-run context → PROGRAM.md
│
└── adapters/
    └── codex/
        ├── README.md               # adapter setup and usage
        ├── context_builder.py      # RunContext from task + artifacts
        ├── item_mapper.py          # ThreadItem → EpisodeEvent
        └── sdk_runner.py           # CodexRunner — full run lifecycle
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- Codex app-server running locally (see `codex/codex-rs/README.md`)
- `codex_app_server` SDK installed from the local Codex repo

### 1. Install the CL layer

```bash
cd cl-layer
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e /path/to/codex/sdk/python   # Codex SDK
```

### 2. Run the tests

```bash
python -m pytest cl-layer/tests/ -v
# 22 tests, all passing, no network dependency
```

### 3. Run a task through the adapter

```python
from adapters.codex.sdk_runner import CodexRunner

runner = CodexRunner(
    episodes_path="data/episodes.jsonl",
    artifacts_dir="data/",
    default_model="o4-mini",
)

episode = runner.run(
    task_prompt="Add a /healthz endpoint to main.py that returns 200 OK.",
    task_id="task-001",
    task_domain="fastapi",
    mode="baseline",          # ephemeral thread — substrate is the only learning signal
    cwd="/path/to/your/repo",
)

print(episode.outcome.status)          # "completed" | "partial" | "failed"
print(episode.outcome.files_touched)
```

### 4. Distil and write learning artifacts

```python
from cl_layer.episode.recorder import EpisodeRecorder
from cl_layer.distill.skills import distill_skills, distill_warnings, render_skills_md
from cl_layer.distill.dreams import summarize_session, render_dreams_md
from cl_layer.distill.program import render_program_md
from pathlib import Path

recorder = EpisodeRecorder("data/episodes.jsonl")
episodes = recorder.load_all()

skills   = distill_skills(episodes)
warnings = distill_warnings(episodes)
summary  = summarize_session(episodes)

Path("data/SKILLS.md").write_text(render_skills_md(skills))
Path("data/DREAMS.md").write_text(render_dreams_md(summary))
Path("data/PROGRAM.md").write_text(render_program_md("fastapi", skills, warnings))
```

### 5. Run again in integrated mode

On the second run the adapter reads `PROGRAM.md` and `SKILLS.md` and injects them as `developer_instructions` before the task executes:

```python
episode = runner.run(
    task_prompt="Add a /readyz endpoint to main.py.",
    task_id="task-002",
    task_domain="fastapi",
    mode="integrated",        # persistent thread, substrate artifacts injected
    cwd="/path/to/your/repo",
)
```

---

## Operating Modes

| Mode | Codex thread | Substrate injection | Use for |
|------|-------------|---------------------|---------|
| `baseline` | `ephemeral=True` — native memory minimized | Off | Controlled experiments; substrate is the only cross-session signal |
| `integrated` | Persistent — native memory coexists | On — `PROGRAM.md` + `SKILLS.md` injected as `developer_instructions` | Daily use where both memory systems coexist |

The distinction matters for research integrity: improvement claims from `baseline` runs cannot be contaminated by Codex's own memory pipeline.

---

## Session Loop

```
Pre-run
  adapter reads PROGRAM.md + SKILLS.md (integrated mode only)
  adapter composes developer_instructions
  adapter calls thread_start(..., developer_instructions=..., ephemeral=baseline)

Run
  adapter calls thread.run(task_prompt)
  Codex executes the task, returning structured ThreadItems
  adapter maps items → normalized EpisodeEvents

Post-run
  substrate writes episode to episodes.jsonl
  distill/skills.py  → updates SKILLS.md
  distill/dreams.py  → updates DREAMS.md
  distill/program.py → updates PROGRAM.md for the next run
```

---

## Persistent Artifacts

| Artifact | Format | Purpose |
|----------|--------|---------|
| `episodes.jsonl` | Append-only JSONL | Primary raw evidence store. One line per episode. |
| `SKILLS.md` | Markdown | Distilled reusable patterns with evidence back-references. |
| `DREAMS.md` | Markdown | Human-readable session summaries: what worked, what failed, replay targets. |
| `PROGRAM.md` | Markdown | Generated context injected at run start. Domain, patterns, failure warnings. |

All four files are human-readable and git-diffable. If a skill entry looks wrong, edit or delete it directly.

---

## Build Phases

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 0 — Architecture lock | Done | `spec.md`, `spec-validation.md`, `plan.md` |
| 1 — Core data plane | Done | `episode/`, `replay/`, 22 passing tests |
| 2 — Distillation | Done | `distill/skills.py`, `distill/dreams.py`, `distill/program.py` |
| 3 — Codex adapter | Done | `adapters/codex/` grounded in the local SDK |
| 4 — Evaluation layer | Planned | `cl_layer/eval/reward.py`, `cl_layer/eval/metrics.py` |
| 5 — Benchmark harness | Planned | Narrow domain suite, baseline vs learned comparison |
| 6 — MCP interface | Planned | `cl_layer/mcp/server.py` — `cl/context`, `cl/log`, `cl/search`, `cl/metrics` |
| 7 — Second agent surface | Planned | Hermes adapter, same core unchanged |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SDK over CLI | The Codex Python SDK returns typed `ThreadItem` objects. Log parsing is brittle; structured items are versioned. |
| File-first persistence | Zero operational overhead, human-readable, git-diffable, portable to any agent that can read a file. |
| Two explicit modes | Mixing Codex native memory into baseline experiments invalidates learning attribution. |
| `reward = None` at record time | Raw observable outcomes come before derived reward. Later evaluation can compute multiple reward framings from the same episode. |
| Rule-based distillation first | Deterministic rules make the learning loop testable and auditable. LLM synthesis is an optimization, not a foundation. |
| No vector search in v1 | Domain keyword filtering is sufficient to prove the substrate claim before adding retrieval infrastructure risk. |
| Narrow benchmark before broad claims | BWT/FWT are only meaningful under controlled exposure conditions. |

---

## Package Documentation

- **CL Layer** — `cl-layer/README.md`: episode model, distillation policy, evaluation design, contribution guidelines, architectural guardrails
- **Codex Adapter** — `adapters/codex/README.md`: setup, mode semantics, item-type mapping table, integration details

---

## License

See `LICENSE`.
