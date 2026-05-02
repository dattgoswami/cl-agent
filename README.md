# cl-agent — Continual-Learning Substrate for Coding Agents

A **Codex-first continual-learning substrate** that attaches to coding agents you already use, records what they tried, distills what worked, and injects that knowledge forward so the agent measurably improves across sessions.

The repo now ships **two complementary learning systems**:

1. **Symbolic substrate (v1)** — episode capture, replay, rule-based distillation, and forward injection. Improves the agent without touching model weights. ([Paper](paper/cl_agent_paper_pub.pdf))
2. **SOAR-style search + SFT pipeline (v2)** — sandboxed sample/refine/repair search, verifier-driven scoring, hindsight relabeling, dataset construction, periodic LoRA fine-tuning of a small student model (Qwen2.5-Coder-3B), GGUF export, and Ollama deployment with a four-mode benchmark runner that feeds promotion gates.

The two systems are kept deliberately separate. The symbolic layer stays inspectable and reversible; the parametric layer captures patterns too subtle for prompt artifacts alone.

> The durable asset is the learning loop, not the runtime shell.

---

## What This Is

This repo ships the core substrate plus thin agent-surface adapters:

| Package | Path | Purpose |
|---------|------|---------|
| **CL Layer** | `cl-layer/` | Core substrate: episode capture, replay buffer, rule-based distillation, evaluation hooks |
| **Codex Adapter** | `adapters/codex/` | Thin adapter that drives Codex via its Python SDK and maps structured `ThreadItem` outputs into normalized episodes |
| **Pi Monorepo Adapter** | `adapters/pi_mono/` | Import-first adapter that maps Pi coding-agent session JSONL into normalized episodes |
| **Hermes Agent Adapter** | `adapters/hermes_agent/` | Import-first adapter that maps Hermes ShareGPT-style batch trajectories into normalized episodes |
| **Aider Adapter** | `adapters/aider/` | Conservative CLI/git adapter that maps Aider subprocess, chat history, and git diff evidence into normalized episodes |
| **SWE-agent Adapter** | `adapters/swe_agent/` | Import-first adapter that maps full SWE-agent `.traj` JSON files into normalized episodes |

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
│  Codex · Pi coding-agent · Hermes · Aider · SWE-agent       │
└──────────────┬───────────────────────────────┬────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐   ┌───────────────────────────┐
│ Codex Adapter            │   │ Pi Adapter                │
│ (adapters/codex/)        │   │ (adapters/pi_mono/)       │
│                          │   │                           │
│ context_builder.py       │   │ session_loader.py         │
│   assembles RunContext   │   │   parses session JSONL    │
│ sdk_runner.py            │   │ item_mapper.py            │
│   drives Codex SDK       │   │   maps entries → events   │
│ item_mapper.py           │   │ context_builder.py        │
│   ThreadItems → events   │   │   builds PiRunContext     │
│                          │   │ runner.py                 │
│                          │   │   thin CLI dispatcher     │
└──────────────┬───────────┘   └───────────────┬───────────┘
               └───────────────┬───────────────┘
                               │
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│ CL Layer   (cl-layer/)                                     │
│                                                            │
│  Symbolic plane (v1):                                      │
│    episode/   — normalized schema + JSONL recorder         │
│    replay/    — domain-heuristic buffer, no embeddings     │
│    distill/   — rule-based skills, dreams, program writers │
│                                                            │
│  Research / data plane (v2):                               │
│    dataset/   — episodes → JSONL splits + manifest         │
│    verify/    — sandboxed runner + git changed-file        │
│    search/    — SOAR sample/refine/repair + sandbox        │
│    train/     — MLX trainer shell + promotion gates        │
│    serve/     — Ollama Modelfile + GGUF export             │
│    eval/      — benchmark loader + 4-mode runner           │
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
│       │   ├── schema.py           # Episode + 25 training-grade fields
│       │   └── recorder.py         # append-only JSONL store
│       ├── replay/
│       │   └── buffer.py           # domain/mode-filtered replay queries
│       ├── distill/
│       │   ├── skills.py           # SKILLS.md
│       │   ├── dreams.py           # DREAMS.md
│       │   └── program.py          # PROGRAM.md
│       │
│       ├── dataset/                # v2: episodes → SFT JSONL
│       │   ├── from_episode.py     # strict, patch_text-required
│       │   ├── filters.py          # verifier / size / hidden-state
│       │   ├── dedup.py            # patch-hash + normalized-text
│       │   ├── splits.py           # leak-proof, grouped by task_id
│       │   ├── render_chat.py      # ChatTemplate + JSONL renderer
│       │   └── builder.py          # build_dataset() + DatasetManifest
│       │
│       ├── verify/                 # v2: sandboxed verification
│       │   ├── base.py             # VerificationResult, CommandRunner
│       │   ├── python_repo.py      # injectable runner + git changed_files
│       │   ├── pytest_runner.py    # pytest step builder
│       │   ├── lint_runner.py      # ruff/flake8 step builder
│       │   ├── typecheck_runner.py # mypy step builder
│       │   └── score.py            # weighted scoring helpers
│       │
│       ├── search/                 # v2: SOAR sample/refine/repair
│       │   ├── sandbox.py          # Sandbox protocol + InMemorySandbox
│       │   ├── sampler.py          # plan → patch via model client
│       │   ├── repair.py           # revised-patch repair (no append)
│       │   ├── selection.py        # weighted score from metadata
│       │   ├── mutation.py         # prompt / file-scope / hunk
│       │   ├── crossover.py        # disjoint-files / non-overlapping
│       │   ├── archive.py          # novelty keys with real deltas
│       │   └── controller.py       # multi-generation soar_loop()
│       │
│       ├── train/                  # v2: trainer backends + promotion
│       │   ├── base.py             # TrainerBackend, TrainConfig
│       │   ├── mlx_backend.py      # MLXTrainerBackend + injectable runner
│       │   ├── unsloth_backend.py  # NotImplemented stub
│       │   ├── promotion.py        # PromotionGate (+5% / -2% / smoke)
│       │   ├── registry.py         # backend registry
│       │   └── export.py           # export manifest helpers
│       │
│       ├── serve/                  # v2: Ollama deployment
│       │   ├── modelfile.py        # valid Modelfile generator
│       │   ├── ollama_create.py    # `ollama create` wrapper
│       │   └── ollama_smoke.py     # smoke prompts via injectable client
│       │
│       └── eval/                   # v2: benchmark + mode comparison
│           ├── benchmark.py        # BenchmarkSuite.from_path (JSON/YAML)
│           ├── runner.py           # 4-mode runner + EvaluationResult
│           └── modes.py            # baseline / symbolic / search / search_sft
│
├── adapters/
│   ├── codex/
│   │   ├── README.md
│   │   ├── context_builder.py
│   │   ├── item_mapper.py
│   │   └── sdk_runner.py
│   ├── pi_mono/
│   │   ├── README.md
│   │   ├── session_loader.py
│   │   ├── item_mapper.py
│   │   ├── context_builder.py
│   │   ├── runner.py
│   │   └── time_utils.py
│   ├── hermes_agent/
│   │   ├── README.md
│   │   ├── trajectory_loader.py
│   │   ├── item_mapper.py
│   │   ├── context_builder.py
│   │   └── runner.py
│   └── aider/
│       ├── README.md
│       ├── log_loader.py
│       ├── item_mapper.py
│       ├── context_builder.py
│       └── runner.py
│
├── benchmarks/                     # v2: benchmark fixtures
│   └── example_suite.json          # 3 tasks across all 3 categories
├── configs/                        # v2: trainer/search/eval configs
└── data/                           # v2: runtime artifacts
    ├── episodes/  datasets/  models/  exports/  reports/
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
# 281+ tests, all passing in <1s  (run pytest -q for current count)
# no network, no MLX, no Torch, no Ollama, no real model calls
```

The full suite imports cleanly without any optional ML dependency installed (MLX, Torch, Transformers, PEFT, TRL, Unsloth, requests, PyYAML are all lazy or test-injected).

### 3a. Run a task through the Codex adapter

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

### 3b. Capture a Pi session through the Pi adapter

The Pi adapter is import-first: run Pi normally, then import the JSONL it wrote.

```bash
# 1. Run Pi on a task (Pi writes a session JSONL automatically)
pi "Add a /healthz endpoint to main.py" --cwd /path/to/your/repo
```

```python
# 2. Import the session into the substrate
from adapters.pi_mono import append_session_episode

episode = append_session_episode(
    "/path/to/pi/session.jsonl",   # consult `pi sessions list` or Pi docs for exact path
    "data/episodes.jsonl",
    task_id="task-001",
    task_domain="fastapi",
    task_description="Add a /healthz endpoint to main.py",
    mode="baseline",
)

print(episode.outcome.status)
print(episode.outcome.files_touched)  # only proven mutations (edit/write)
```

See `adapters/pi_mono/README.md` for branch selection, integrated mode, and the full loop.

### 3c. Capture Hermes batch trajectories

The Hermes adapter is import-first: run Hermes batch generation separately, then
import its `trajectories.jsonl`.

```python
from adapters.hermes_agent import append_trajectory_episodes

episodes = append_trajectory_episodes(
    "/path/to/hermes/run/trajectories.jsonl",
    "data/episodes.jsonl",
    task_id_prefix="hermes-task",
    task_domain="python",
    mode="baseline",
)

print(len(episodes))
```

See `adapters/hermes_agent/README.md` for supported formats, mode semantics,
Hermes native-memory attribution notes, and current limitations.

### 3d. Run and capture Aider through the Aider adapter

The Aider adapter is conservative: it runs Aider through an injectable CLI boundary and records subprocess output, run-specific chat history, and git evidence before/after the run.

```python
from adapters.aider import AiderRunner

runner = AiderRunner(
    "data/episodes.jsonl",
    artifacts_dir="data/",
)

episode = runner.run(
    "Fix the failing pytest case.",
    task_id="task-001",
    task_domain="python",
    mode="baseline",
    cwd="/path/to/your/git/repo",
    model="sonnet",
    test_cmd="pytest -q",
    auto_test=True,
)

print(episode.outcome.status)
print(episode.outcome.files_touched)
print(episode.patch_hash)
```

By default the runner disables Aider auto-commits and dirty pre-edit commits so the adapter can capture the resulting working-tree patch. It also disables restored chat history and uses per-run capture files under `aider-captures/<run_id>/`.

### 3e. Capture SWE-agent trajectories

The SWE-agent adapter imports full `.traj` JSON files after a `sweagent run` or
`sweagent run-batch` execution. It does not run Docker, model calls, or
SWE-bench evaluation during import.

```python
from adapters.swe_agent import append_trajectory_episode

episode = append_trajectory_episode(
    "/path/to/trajectories/run/pydicom__pydicom-1458.traj",
    "data/episodes.jsonl",
    task_domain="swe_bench",
    mode="baseline",
)

print(episode.outcome.status)
print(episode.outcome.files_touched)
print(episode.patch_hash)
```

See `adapters/swe_agent/README.md` for supported `.traj` variants, config
overlay semantics, mapping notes, and deferred live-run work.

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

On the second run each adapter reads `PROGRAM.md` and `SKILLS.md` and injects them before the task executes.

**Codex** — injected as `developer_instructions` on the SDK thread:

```python
episode = runner.run(
    task_prompt="Add a /readyz endpoint to main.py.",
    task_id="task-002",
    task_domain="fastapi",
    mode="integrated",        # persistent thread, substrate artifacts injected as developer_instructions
    cwd="/path/to/your/repo",
)
```

**Pi** — injected via `--append-system-prompt` on the CLI command:

```python
from adapters.pi_mono import PiCliRunner

runner = PiCliRunner(artifacts_dir="data/")
result = runner.run(
    "Add a /readyz endpoint to main.py.",
    mode="integrated",        # injects PROGRAM.md + SKILLS.md via --append-system-prompt
    cwd="/path/to/your/repo",
)
# Then import the session JSONL Pi wrote and repeat from step 2 of section 3b.
```

**Aider** — injected into the one-shot `--message` prompt:

```python
from adapters.aider import AiderRunner

runner = AiderRunner("data/episodes.jsonl", artifacts_dir="data/")
episode = runner.run(
    "Add a /readyz endpoint to main.py.",
    task_id="task-002",
    task_domain="fastapi",
    mode="integrated",        # prepends PROGRAM.md + SKILLS.md to the Aider message
    cwd="/path/to/your/repo",
)
```

---

## Operating Modes

| Mode | Codex | Pi | Aider | Substrate injection | Use for |
|------|-------|----|-------|---------------------|---------|
| `baseline` | `ephemeral=True` — native memory minimized | `--no-session` — ephemeral run | run-specific history files; no restored chat | Off | Controlled experiments; substrate is the only cross-session signal |
| `integrated` | Persistent — native memory coexists | Normal session | run-specific history files; CL artifacts prepended to prompt | On — `PROGRAM.md` + `SKILLS.md` injected (`developer_instructions` for Codex; `--append-system-prompt` for Pi; `--message` for Aider) | Daily use where both memory systems coexist |

The distinction matters for research integrity: improvement claims from `baseline` runs cannot be contaminated by agent-native memory pipelines.

---

## Session Loop

### Codex (live driving)

```
Pre-run
  adapter reads PROGRAM.md + SKILLS.md (integrated mode only)
  adapter composes developer_instructions
  adapter calls thread_start(..., developer_instructions=..., ephemeral=baseline)

Run
  adapter calls thread.run(task_prompt)
  Codex executes the task, returning structured ThreadItems
  adapter maps ThreadItems → normalized EpisodeEvents

Post-run
  substrate writes episode to episodes.jsonl
  distill/skills.py  → updates SKILLS.md
  distill/dreams.py  → updates DREAMS.md
  distill/program.py → updates PROGRAM.md for the next run
```

### Pi (import-first)

```
Pre-run (integrated mode only)
  ContextBuilder reads PROGRAM.md + SKILLS.md
  PiCliRunner passes --append-system-prompt <artifacts> to the Pi CLI

Run
  Pi executes the task externally, writing an append-only session JSONL file

Post-run
  import_session_jsonl / append_session_episode reads the JSONL
  session_loader selects the active branch (last leaf → root walk)
  item_mapper maps entries → normalized EpisodeEvents + EpisodeOutcome
  substrate writes episode to episodes.jsonl
  distill/skills.py  → updates SKILLS.md
  distill/dreams.py  → updates DREAMS.md
  distill/program.py → updates PROGRAM.md for the next run
```

### Aider (CLI/git capture)

```
Pre-run
  ContextBuilder reads PROGRAM.md + SKILLS.md (integrated mode only)
  AiderRunner creates a run-specific capture directory
  adapter snapshots git HEAD/status/diff before execution
  adapter runs aider --message <task> with safe capture flags

Run
  Aider executes externally
  adapter captures subprocess stdout/stderr
  Aider writes chat, input, and LLM history files under aider-captures/<run_id>/

Post-run
  adapter snapshots git HEAD/status/diff after execution
  log_loader parses Aider chat history into coarse user/assistant/tool messages
  item_mapper maps subprocess, command-output, chat, and git-diff evidence
  substrate writes episode to episodes.jsonl
  distill/skills.py  → updates SKILLS.md
  distill/dreams.py  → updates DREAMS.md
  distill/program.py → updates PROGRAM.md for the next run
```

The Aider adapter intentionally avoids parsing assistant prose as edit evidence. `files_touched`, `patch_text`, and `patch_hash` come from git. If the worktree was already dirty before the run, the adapter avoids attributing pre-existing dirty paths or diffs to Aider.

### SWE-agent (trajectory import)

```
Pre-run
  ContextBuilder can write an explicit SWE-agent config overlay
  baseline clears demonstrations and injects no CL artifacts
  integrated injects PROGRAM.md + SKILLS.md through strategy_template

Run
  SWE-agent executes externally and writes <instance_id>.traj

Post-run
  load_trajectory reads the full .traj JSON plus adjacent config when present
  item_mapper maps history, thought/action/observation turns, commands, edits, and submission diffs
  info.submission becomes patch_text + patch_hash when present
  substrate writes episode to episodes.jsonl
  distill/skills.py  → updates SKILLS.md
  distill/dreams.py  → updates DREAMS.md
  distill/program.py → updates PROGRAM.md for the next run
```

SWE-bench pass/fail is usually produced outside the `.traj` file. The adapter
records `benchmark_result` only when the trajectory `info` object explicitly
contains resolved or benchmark output.

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

### v1 — Symbolic substrate

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 0 — Architecture lock | Done | `spec.md`, `spec-validation.md`, `plan.md` |
| 1 — Core data plane | Done | `episode/`, `replay/` |
| 2 — Distillation | Done | `distill/skills.py`, `distill/dreams.py`, `distill/program.py` |
| 3 — Codex adapter | Done | `adapters/codex/` — live SDK driving, `ThreadItem` mapping |
| 3b — Pi adapter | Done | `adapters/pi_mono/` — import-first JSONL capture, branch selection, integrated-mode injection |
| 3c — Hermes adapter | Done | `adapters/hermes_agent/` — import-first batch trajectory capture, XML tool-call parsing, native-memory attribution docs |
| 3d — Aider adapter | Done | `adapters/aider/` — CLI subprocess capture, run-specific chat history parsing, safe git diff attribution |
| 3e — SWE-agent adapter | Done | `adapters/swe_agent/` — full `.traj` import, submission patch capture, explicit config overlay preview |

### v2 — SOAR-style search + SFT pipeline

The v2 plane was added on top of v1 to close the "second half of the loop": search over candidate solutions, verifier-driven refinement, hindsight relabeling, periodic SFT, and Ollama deployment.

| Phase | Status | Deliverable |
|-------|--------|-------------|
| Phase 0 — Instrumentation | Done | 25 training-grade fields on `Episode`, 9 new event kinds, back-compat tests |
| Phase 1 — Dataset builder | Done | `dataset/builder.py` — episodes → JSONL splits + manifest, leak-proof grouping by `task_id`, patch-hash dedup |
| Phase 1 — Chat template + Modelfile | Done | One `ChatTemplate` shared by `dataset/render_chat.py` and `serve/modelfile.py`; trainer↔Ollama parity guaranteed by golden test |
| Phase 1 — Verifier framework | Done | Injectable `CommandRunner`, `task_id` threading, best-effort `git status --porcelain` for `changed_files` |
| Phase 1 — MLX trainer shell | Done | `MLXTrainerBackend` calls injectable runner for `train_lora`/`fuse_model`/`convert_gguf`; `output_dir` separate from `dataset_dir` |
| Phase 1 — Ollama serve | Done | Valid Modelfile with `SYSTEM` / `TEMPLATE` triple-quote blocks and Go-template variables; mocked smoke tests |
| Phase 2 — SOAR controller | Done | Sandboxed multi-generation `soar_loop()`; verifier sees repo paths only; `max_generations` and `max_candidates_total` enforced; weighted scoring from real verifier inputs; novelty archive with real score deltas |
| Phase 1 — Eval + promotion | Done | `BenchmarkSuite.from_path()` (JSON, optional YAML); 4-mode runner (`baseline` / `symbolic` / `search` / `search_sft`) → `EvaluationResult` → `PromotionGate` (+5% lift / −2% regression / smoke / latency) |
| Integration smoke | Done | `tests/test_integration_learning_path.py` walks episodes → dataset → mocked train/fuse/gguf → Modelfile in <50 ms |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Structured data over text scraping | Each adapter consumes the most structured interface the agent exposes: typed SDK objects for Codex (`ThreadItem`), session JSONL for Pi, ShareGPT-style batch trajectory JSONL for Hermes, git/subprocess evidence for Aider, and full `.traj` JSON for SWE-agent. Text scraping is limited to coarse chat/tool messages and is never the authority for file changes. |
| File-first persistence | Zero operational overhead, human-readable, git-diffable, portable to any agent that can read a file. |
| Two explicit modes | Mixing agent-native memory into baseline experiments invalidates learning attribution. The `baseline` / `integrated` distinction holds across adapters regardless of how each agent implements its own memory. |
| `reward = None` at record time | Raw observable outcomes come before derived reward. Later evaluation can compute multiple reward framings from the same episode. |
| Rule-based distillation first | Deterministic rules make the learning loop testable and auditable. LLM synthesis is an optimization, not a foundation. |
| No vector search in v1 | Domain keyword filtering is sufficient to prove the substrate claim before adding retrieval infrastructure risk. |
| Narrow benchmark before broad claims | BWT/FWT are only meaningful under controlled exposure conditions. |

---

## Package Documentation

- **CL Layer** — `cl-layer/README.md`: episode model, distillation policy, evaluation design, contribution guidelines, architectural guardrails
- **Codex Adapter** — `adapters/codex/README.md`: setup, mode semantics, item-type mapping table, integration details
- **Pi Monorepo Adapter** — `adapters/pi_mono/README.md`: Pi session JSONL importer, mode semantics, event mapping, live-run boundary
- **Hermes Agent Adapter** — `adapters/hermes_agent/README.md`: Hermes batch trajectory importer, mode semantics, event mapping, native-memory separation
- **Aider Adapter** — `adapters/aider/README.md`: CLI/git capture, safe defaults around auto-commits, chat-history parsing, event mapping, limitations
- **SWE-agent Adapter** — `adapters/swe_agent/README.md`: full `.traj` importer, mode semantics, event mapping, config overlay preview, benchmark notes

---

## References

The v1 substrate paper is at [`paper/cl_agent_paper_pub.pdf`](paper/cl_agent_paper_pub.pdf).

### External research and tools the v2 layer draws on

**SOAR** — the LLM-guided sample-refine-search framework for ARC tasks. The v2 search loop adapts SOAR's ARC loop (sample → execute → verify → select → refine → relabel → curate → fine-tune → evaluate → promote) to the software-engineering setting.
- Pourcel et al., *"SOAR: Self-Improving Optimization Agent via Reinforcement learning"* — paper on arXiv: <https://arxiv.org/abs/2507.14172>
- PMLR 2025 version: <https://proceedings.mlr.press/v267/pourcel25a.html>
- Reference implementation: <https://github.com/flowersteam/SOAR>

**MLX-LM** — primary local trainer target for Apple Silicon.
- <https://github.com/ml-explore/mlx-lm>
- LoRA recipe: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>

**Unsloth** — secondary trainer backend for NVIDIA/Linux experiments.
- <https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements>
- Saving to Ollama: <https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama>
- Saving to GGUF: <https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf>

**Ollama** — runtime target. The `serve/modelfile.py` generator emits a `SYSTEM`/`TEMPLATE` block matching the trainer's ChatML render byte-for-byte.
- Import docs: <https://docs.ollama.com/import>
- Modelfile reference: <https://docs.ollama.com/modelfile>

**Qwen models** — student / teacher candidates.
- `Qwen/Qwen2.5-Coder-3B-Instruct` (primary student): <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct>
- `Qwen/Qwen2.5-Coder-7B-Instruct` (Phase 2 student): <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct>
- `Qwen/Qwen3-Coder-30B-A3B-Instruct` (potential teacher): <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>
- Qwen3-Coder blog: <https://qwenlm.github.io/blog/qwen3-coder/>

---

## License

See `LICENSE`.
