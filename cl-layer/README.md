# Continual-Learning Agent Substrate

A **Codex-first continual-learning substrate for coding agents**.

This project is not a new coding agent. It is a thin learning loop that
attaches to agents you already use, records what they tried, distills what
worked, and injects that knowledge forward so the agent measurably improves
across sessions without model fine-tuning.

> The durable asset is the learning loop, not the runtime shell.

---

## Who Benefits

| Audience | How it helps |
|----------|-------------|
| **Engineering teams using AI coding tools** | Stops the agent from repeating the same mistakes in the same repo. Turns scattered session usage into durable, team-visible institutional memory. |
| **Platform / DX engineers** | Provides a structured evidence layer between the agent and your CI/CD loop. Every completed or failed task produces a normalized episode that can feed metrics dashboards or post-mortems. |
| **Researchers studying continual learning in LLMs** | Offers a real-world episodic testbed. BWT/FWT-style metrics are built in, but tied to a proper narrow-domain benchmark protocol rather than anecdote. |
| **Solo engineers on complex long-running codebases** | The agent accumulates repo-specific knowledge — what breaks CI, which patterns are safe, how tests are structured — without you having to re-explain it every session. |
| **Teams moving toward agentic / autonomous workflows** | Provides a shared failure-warning and replay layer that makes multi-agent or overnight runs safer without requiring you to build RL infrastructure. |

The common thread: **any team that wants their coding agent to get better at
their codebase over time, without fine-tuning a model or replacing their
current tooling.**

---

## Why This Exists: Ecosystem Survey

This project emerged from a survey of 30+ agent, memory, and orchestration
repos that already exist locally
(`codex`, `hermes-agent`, `opencode`, `langchain`, `langgraph`, `mastra`,
`Acontext`, `Memento`, `honcho`, `Papr Memory`, `langfuse`, `langsmith-sdk`,
`dev-agent`, `autoresearch`, `OpenClaw-RL`, and more).

The survey found that the ecosystem is **strong in pieces but weak in the
seams**. Seven specific gaps remain unaddressed across all of them:

### 1. No framework-agnostic continual-learning substrate

> "Several repos have memory, skills, RL, or self-improvement, but there is
> no obvious plug-any-agent-into-it layer."

`Acontext` gives you inspectable skill files. `Memento` gives you
case-based reasoning. `honcho` gives you user-entity memory. `OpenClaw-RL`
gives you an async RL framework. None of them give you a portable layer that
records what a coding agent tried, whether it worked, and surfaces that
knowledge on the next run — regardless of which agent you use.

### 2. No shared run or trajectory format

Prompts, tool calls, code diffs, test outcomes, reviewer feedback, and final
task results are not normalized across `codex`, `opencode`, `hermes-agent`,
`mastra`, or `langgraph`. Every tool produces its own log format. There is no
common schema for "what the agent did in this session."

### 3. No real coding-agent long-term memory

`dev-agent` helps with code retrieval from the current repo. The memory
repos help with recall of user preferences and entities. But there is no
standard system that learns from **repeated coding sessions**, PR review
feedback, test failures, and repository history together and makes that
learning available on the next run.

### 4. No memory interoperability

`Acontext` skills, `honcho` entity memory, `Papr` graph memory, `Memento`
cases, `LangGraph` memory, and `hermes-agent` skill concepts do not cleanly
interchange. A pattern learned in one agent surface cannot be trivially
replayed in another.

### 5. No safe promotion and rollback of learned behavior

Few obvious signs of a shared pattern for deduplication, forgetting,
versioning, quality gates, and rollback after a bad memory update or weak
skill synthesis. Systems accumulate knowledge but have no standard way to
demote, correct, or expire it.

### 6. No cross-framework eval and replay

`langfuse` and `langsmith-sdk` can observe systems, but a common benchmark
that proves **"this agent got better over time because of memory/learning"**
across coding, browser, and research tasks still looks missing.

### 7. No portable multi-agent memory-sharing model

Many systems support subagents, but there is no clear standard for how
agents should share learned skills, user preferences, and execution lessons
safely across a team or across runs.

### The conclusion from the survey

> "You do not need to build the entire stack again. The high-leverage move is
> to build the missing connective tissue: a continual-learning layer for
> existing agents, especially coding agents, with portable memory, measurable
> improvement, and easy adapters into the tools that already exist."

This project is that connective tissue. It does **not** replace `codex`,
`langfuse`, `Acontext`, or `dev-agent` — it is the layer that makes them
work together across sessions.

---

## What This Is Not

- Not a new agent shell or TUI
- Not a replacement for Codex native memory
- Not a vector database or memory service
- Not a general-purpose agent orchestration framework
- Not a full continual-RL training system in v1
- Not a proof of broad cross-domain coding-agent generalization yet

---

## The Four Pillars

Every piece of this project serves one of four responsibilities:

```
1. Episode capture   — record what the agent did, with raw evidence
2. Replay            — surface relevant prior failures and successes
3. Distillation      — extract inspectable, durable knowledge artifacts
4. Evaluation        — measure whether the agent is actually improving
```

Removing any one pillar degrades the project into something else:
without capture it is just prompting; without replay just logging;
without distillation just an event store; without metrics just a memory
feature.

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

The core learning artifacts are plain files. Any agent that can read a
file can benefit from them. The adapter is the only surface-specific code.

---

## Repository Layout

```
continual-learning-agent/
│
├── cl-layer/                       # this package — push to GitHub
│   ├── pyproject.toml
│   ├── .gitignore
│   └── cl_layer/
│       ├── episode/
│       │   ├── schema.py           # EpisodeEvent, EpisodeOutcome, Episode
│       │   └── recorder.py         # append-only JSONL store
│       ├── replay/
│       │   └── buffer.py           # domain/mode-filtered replay queries
│       ├── distill/
│       │   ├── skills.py           # rule-based skill candidates + SKILLS.md
│       │   ├── dreams.py           # session summary + DREAMS.md
│       │   └── program.py          # next-run context + PROGRAM.md
│       └── eval/                   # Phase 4 — reward + operational metrics
│
├── adapters/
│   └── codex/
│       ├── README.md               # setup and usage
│       ├── context_builder.py      # RunContext from task + artifacts
│       ├── item_mapper.py          # ThreadItem → EpisodeEvent
│       └── sdk_runner.py           # CodexRunner — full run lifecycle
│
├── ideas-plan-spec/                # design docs (source of truth)
│   ├── spec.md
│   ├── spec-validation.md
│   ├── plan.md
│   ├── codex-action-plan.md
│   └── thoughts.md
│
└── codex/                          # local Codex repo (agent surface)
    └── sdk/python/                 # codex_app_server SDK
```

---

## Codex Integration: What We Learned From the Real SDK

The initial spec assumed a CLI-based Codex flow — parsing log output from a
`codex run` command. Inspecting the actual local repo changed several
implementation decisions.

### Codex has a native memory pipeline

Codex is **not stateless per run**. It ships a two-phase memory pipeline
(`codex/codex-rs/core/src/memories/README.md`):

- **Phase 1** — extracts a structured memory from each completed rollout and
  stores it in a local state DB
- **Phase 2** — consolidates phase-1 outputs into on-disk artifacts
  (`raw_memories.md`, `rollout_summaries/`) via an internal consolidation
  sub-agent

This pipeline runs at session startup for non-ephemeral threads. It is the
reason the substrate must define two explicit modes:

| Mode | `ephemeral` flag | Codex native memory | Use for |
|------|-----------------|---------------------|---------|
| `baseline` | `True` | Disabled | Controlled experiments — substrate is the only learning signal |
| `integrated` | `False` (default) | Active | Daily use — native memory and substrate coexist |

### The correct integration path is the Python SDK, not CLI

The local SDK (`codex/sdk/python/`) exposes:

```python
from codex_app_server import Codex

with Codex() as codex:
    thread = codex.thread_start(
        cwd="/path/to/repo",
        developer_instructions="...",   # substrate artifacts injected here
        ephemeral=True,                 # baseline mode
        model="o4-mini",
    )
    result = thread.run("Add a /healthz endpoint to main.py.")
    # result.items  → list[ThreadItem]  — typed, structured
    # result.final_response → str | None
```

The previous spec assumed `codex run --config AGENTS.md --tasks TASKS.md`.
That flag shape does not exist in the local repo. The SDK path is the only
verified integration surface.

### ThreadItems are typed — no log parsing needed

`result.items` is `list[ThreadItem]`, a Pydantic discriminated union
(`codex/sdk/python/src/codex_app_server/generated/v2_all.py`).
The adapter maps these directly by `item.type`:

| `item.type` | Fields used | Maps to `EpisodeEvent.kind` |
|-------------|-------------|----------------------------|
| `commandExecution` | `.command`, `.exit_code`, `.aggregated_output`, `.duration_ms` | `command_execution` |
| `fileChange` | `.changes[*].path`, `.status` | `file_change` |
| `mcpToolCall` | `.tool`, `.server`, `.status`, `.duration_ms` | `mcp_tool_call` |
| `agentMessage` | `.text`, `.phase` | `agent_message` |
| all others | — | skipped |

Key verified model details:
- `Thread.id` — the canonical thread identifier (not `thread_id`)
- `FileUpdateChange.path` — the file path field on each change entry
- `developer_instructions` — the `thread_start()` parameter used to inject
  substrate context; files are **not** auto-consumed from disk

### `PROGRAM.md` and `SKILLS.md` require explicit injection

Placing these files in the working directory does not make them visible to
Codex. The `ContextBuilder` must read them and pass their content as
`developer_instructions`. This is enforced in `context_builder.py` and is
only active in `integrated` mode.

---

## Key Design Decisions

### 1. Codex SDK over CLI scraping

The local Codex repo ships a Python SDK with `Codex().thread_start(...)`,
`thread.run(...)`, and typed `ThreadItem` returns
(`CommandExecutionThreadItem`, `FileChangeThreadItem`, `McpToolCallThreadItem`,
`AgentMessageThreadItem`).

The adapter uses those structured items as its primary data source. It never
parses terminal output and never assumes CLI flags that were not verified
against the real SDK.

### 2. File-first persistence

`episodes.jsonl` for episodes, Markdown for distilled artifacts.

Reasons: zero operational overhead, human-readable, git-diffable,
portable to any agent that can read a file. DuckDB or a proper store can be
layered on later once the research claim is proven on file-based replay.

### 3. Two explicit operating modes

| Mode | Codex thread | Substrate context injection | When to use |
|------|-------------|----------------------------|-------------|
| `baseline` | `ephemeral=True` — native memory minimized | Off — substrate is the only cross-session signal | Controlled experiments, benchmarking |
| `integrated` | Persistent — native memory coexists | On — `PROGRAM.md` and `SKILLS.md` injected as `developer_instructions` | Daily practical usage |

This distinction matters for research integrity: improvement claims from
baseline-mode runs cannot be contaminated by Codex's own memory pipeline.

### 4. Raw outcomes before derived reward

Every episode records observable state first:
- commands executed and their exit codes
- files changed
- MCP tool calls
- agent messages
- tests passed / not passed
- final response

`reward` is an optional float that is `None` at record time and filled by an
evaluator later. Nothing is collapsed to a scalar at capture time.

### 5. Rule-based distillation before LLM synthesis

v1 distillation is deterministic:

- domain with ≥ 2 `completed` episodes → `SkillCandidate`
- domain with ≥ 2 failure episodes → warning string
- session episodes → `DREAMS.md` summary
- skills + warnings → `PROGRAM.md` context for the next run

LLM-based skill synthesis is explicitly deferred. If the deterministic loop
does not work, adding an LLM will only hide the design problem.

### 6. Context injection is the adapter's job

`PROGRAM.md`, `SKILLS.md`, and `DREAMS.md` are **not** assumed to be
auto-consumed by Codex. The `ContextBuilder` reads them and passes their
content as `developer_instructions` in `thread_start(...)`. Injection is
always explicit and testable.

### 7. MCP is Phase 6, not Phase 1

The Codex adapter drives the run externally and can inject context directly.
MCP becomes relevant for portability and agent self-service retrieval after
the direct Codex path is proven. Adding it earlier would create abstraction
before the first real run exists.

### 8. Narrow benchmark before broad claims

BWT/FWT-style metrics are implemented, but they are only meaningful under
a benchmark protocol with:
- known task domains (e.g., repeated FastAPI endpoint tasks)
- a true cold-start baseline
- a fixed exposure order

Ad hoc improvements are reported as operational metrics only.

---

## Quickstart

### Prerequisites

- Python 3.10+
- Codex app-server running locally (see `codex/codex-rs/README.md`)
- `codex_app_server` SDK installed from the local repo

### 1. Install the CL layer

```bash
cd continual-learning-agent/cl-layer
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ../codex/sdk/python   # Codex SDK
```

### 2. Run the tests

```bash
python -m pytest tests/ -v
# 22 tests, all passing, no network dependency
```

### 3. Run a task through the Codex adapter

```python
from adapters.codex.sdk_runner import CodexRunner

runner = CodexRunner(
    episodes_path="data/episodes.jsonl",
    artifacts_dir="data/",       # substrate artifacts live here
    default_model="o4-mini",
)

episode = runner.run(
    task_prompt="Add a /healthz endpoint to main.py that returns 200 OK.",
    task_id="task-001",
    task_domain="fastapi",
    mode="baseline",             # ephemeral thread, no context injection
    cwd="/path/to/your/repo",
)

print(episode.outcome.status)        # "completed" | "partial" | "failed"
print(episode.outcome.files_touched)
```

### 4. Distil and write the learning artifacts

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

On the second run the adapter injects `PROGRAM.md` and `SKILLS.md` as
`developer_instructions` before the task starts:

```python
episode = runner.run(
    task_prompt="Add a /readyz endpoint to main.py.",
    task_id="task-002",
    task_domain="fastapi",
    mode="integrated",           # persistent thread, artifacts injected
    cwd="/path/to/your/repo",
)
```

After enough runs in the same domain, `SKILLS.md` will contain evidence-backed
patterns and `DREAMS.md` will record what worked, what failed, and which
episodes are recommended for replay.

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
  eval/metrics.py    → computes operational metrics (Phase 4)
```

---

## Persistent Artifacts

| Artifact | Format | Purpose |
|----------|--------|---------|
| `episodes.jsonl` | Append-only JSONL | Primary raw evidence store. One line per episode. |
| `SKILLS.md` | Markdown | Distilled reusable patterns with evidence back-references. |
| `DREAMS.md` | Markdown | Human-readable session summaries: what worked, what failed, replay targets. |
| `PROGRAM.md` | Markdown | Generated context injected at run start. Domain, patterns, failure warnings. |

All four files are designed to be read by humans and diffed in version control.
They are not opaque blobs. If a skill entry looks wrong, you can edit or delete
it directly.

---

## Episode Model

```python
@dataclass
class EpisodeEvent:
    kind: Literal["command_execution", "file_change",
                  "mcp_tool_call", "agent_message", "evaluation_result"]
    timestamp: datetime
    payload: dict                # raw structured data from the Codex ThreadItem

@dataclass
class EpisodeOutcome:
    status: Literal["completed", "partial", "failed", "escalated"]
    tests_passed: bool | None    # None until evaluator fills it
    verification_summary: str | None
    escalation_reason: str | None
    files_touched: list[str]
    final_response: str | None

@dataclass
class Episode:
    episode_id: str
    run_id: str
    thread_id: str | None        # Codex Thread.id
    task_id: str
    task_description: str
    task_domain: str
    agent_surface: str           # "codex", "hermes", ...
    mode: Literal["baseline", "integrated"]
    started_at: datetime
    ended_at: datetime
    events: list[EpisodeEvent]
    outcome: EpisodeOutcome
    reward: float | None         # None at record time; evaluator-derived later
```

Raw outcomes are always preserved. Reward is derived, never the primary ground truth.

---

## Evaluation (Phase 4)

Evaluation separates operational metrics from research metrics.

**Operational metrics** (computable from any run):

- Task completion rate
- Escalation rate
- Failure recurrence rate
- Repair attempts per solved task
- Command/tool count per task
- Runtime per task
- Cost per solved task (when available)

**Research metrics** (only meaningful under a controlled benchmark protocol):

- Forward Transfer (FWT): does prior experience on domain A help later tasks
  in domain A?
- Backward Transfer (BWT): does new learning degrade performance on previously
  solved tasks?

Research metrics require:
1. A known task domain with fixed tasks
2. A true cold-start baseline run
3. A known exposure order

They must not be presented from ad hoc runs.

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

## Contribution Guidelines

### Principles before code

1. **Pillar integrity**: every contribution should serve one of the four pillars
   (capture, replay, distillation, evaluation). If it does not, it is likely
   out of scope for this project.

2. **No new agent shells**: do not add a new agent runtime, TUI, or CLI wrapper.
   The adapter pattern keeps agent surfaces swappable without polluting the
   core.

3. **No premature abstraction**: three similar functions are better than a
   generalized base class that does not yet have three real callers.

4. **Defer infrastructure**: do not add a database, vector store, or message
   queue before the flat-file version has been proven under a real benchmark.

5. **Keep artifacts human-readable**: every output the substrate writes
   (`SKILLS.md`, `DREAMS.md`, `PROGRAM.md`, `episodes.jsonl`) must remain
   readable and editable by a human without tooling.

### Adding a new agent surface

1. Create `adapters/<surface>/`.
2. Implement `ContextBuilder`, `ItemMapper`, and a runner class.
3. Map the surface's native output format to `EpisodeEvent` using duck-typing
   on whatever discriminator the surface provides.
4. Do **not** change `cl_layer/episode/schema.py` — the schema is
   surface-agnostic by design.
5. Add a `README.md` in the adapter directory documenting setup, mode
   semantics, and which item types are mapped.

The acceptance test: a run through the new adapter should produce the same
`episodes.jsonl` schema, and the same `distill/` functions should work on the
combined history without modification.

### Adding a new distillation rule

Rules live in `cl_layer/distill/skills.py`.

- Rules must be deterministic and testable with synthetic episodes.
- Each skill candidate must preserve its `evidence_episode_ids` so the source
  is always traceable.
- Do not synthesize skills with an LLM until the rule-based pipeline is
  stable and the benchmark shows it helps.

### Adding evaluation metrics

Metrics live in `cl_layer/eval/` (Phase 4).

- Operational metrics must be computable from a single run's episode.
- Research metrics must be guarded behind an explicit benchmark-protocol flag.
- Do not introduce `R_cold` or transfer-claim constants without a real
  cold-start experimental design behind them.

### Testing

- All `cl_layer/` modules must have tests that run without a live Codex
  instance. The substrate core has no network dependencies.
- Adapter tests may mock the `codex_app_server` SDK. Prefer lightweight
  dataclass stubs that match the real `ThreadItem.type` discriminator.
- Run tests before submitting:
  ```bash
  python -m pytest tests/ -v
  ```

### Commit hygiene

- One logical change per commit.
- Commit messages explain *why*, not just *what*.
- Do not commit `.venv/`, `.pytest_cache/`, or generated runtime artifacts
  (`episodes.jsonl`, `SKILLS.md`, `DREAMS.md`, `PROGRAM.md`).

---

## Architectural Guardrails

These decisions are load-bearing. Do not reverse them without a benchmark
showing it is necessary.

| Guardrail | Rationale |
|-----------|-----------|
| Structured ThreadItems over log parsing | Log parsing is brittle and ties the substrate to a specific output format. The SDK gives typed, versioned models. |
| Baseline mode uses ephemeral threads | Mixing Codex native memory into baseline experiments invalidates the learning attribution. |
| Explicit context injection, never auto-consumed files | `PROGRAM.md` placed in a directory does not become Codex input on its own. The adapter must pass it as `developer_instructions`. |
| `reward` is `None` at record time | Collapsing raw outcomes to a scalar at capture time discards evidence. Later evaluation can compute multiple reward framings from the same raw episode. |
| Rule-based distillation first | Deterministic rules make the learning loop testable and auditable. LLM synthesis is an optimization, not a foundation. |
| No vector search in v1 | Domain keyword filtering is sufficient to prove the substrate claim. Vector search adds infrastructure risk and retrieval unpredictability before the signal is proven. |
| Narrow benchmark before broad claims | BWT/FWT are only meaningful under controlled exposure conditions. Anecdotal improvement is not a research result. |

---

## Deferred Work

The following are explicitly out of scope until the Codex-first path and a
narrow benchmark are working:

- LLM-based skill synthesis
- Semantic / vector retrieval
- OpenClaw or OpenClaw-RL integration
- Multi-agent coordination
- Custom dashboard or UI
- Hermes adapter (Phase 7)
- Policy-gradient or gradient-update language in any implementation doc
- Cross-domain generalization claims

---

## Research Framing

The precise claim this project can support in v1:

> A thin substrate built from episode recording, replay, rule-based skill
> distillation, and context injection can improve coding-agent performance
> across repeated tasks in a narrow domain without model fine-tuning.

That claim is only credible with:
- a controlled narrow benchmark (e.g., repeated FastAPI endpoint tasks)
- true cold-start baselines
- explicit domain boundaries and exposure order

Broader claims require broader experimental evidence.

---

## Relationship to Existing Tools

| Tool | Role in this project |
|------|---------------------|
| `codex` (local) | First agent surface. Driven via `codex_app_server` Python SDK. |
| `autoresearch` | Design pattern reference: overnight loop, keep/reject discipline, `program.md` pattern. Not a direct dependency. |
| `Acontext` | Inspiration for Markdown-first skill file format and portable, inspectable memory. |
| `Memento` | Reference for case-based reasoning and learning from trajectory success/failure. |
| `langfuse` / `langsmith-sdk` | Future observability integration (Phase 4+). Proving whether the learning layer actually works. |
| `dev-agent` | Future codebase retrieval complement — code search to pair with episode-based replay. |
| `hermes-agent` | Second agent surface (Phase 7). Already learning-oriented; shares the thesis. |
| `DuckDB` | Optional future analytics layer over `episodes.jsonl`. Not needed for v1. |
| `OpenClaw-RL` | Future research reference for online RL from conversation feedback, after v1 baseline is solid. |

---

## License

See `LICENSE`.
