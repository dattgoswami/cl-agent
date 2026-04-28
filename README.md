# cl-agent — Continual-Learning Substrate for Coding Agents

A **Codex-first continual-learning substrate** that attaches to coding agents you already use, records what they tried, distills what worked, and injects that knowledge forward so the agent measurably improves across sessions.

The repo now ships **two complementary learning systems**:

1. **Symbolic substrate (v1)** — episode capture, replay, rule-based distillation, and forward injection. Improves the agent without touching model weights. ([Paper](paper/cl_agent_paper_pub.pdf))
2. **SOAR-style search + SFT pipeline (v2)** — sandboxed sample/refine/repair search, verifier-driven scoring, hindsight relabeling, dataset construction, periodic LoRA fine-tuning of a small student model (Qwen2.5-Coder-3B), GGUF export, and Ollama deployment with a four-mode benchmark runner that feeds promotion gates.

The two systems are kept deliberately separate. The symbolic layer stays inspectable and reversible; the parametric layer captures patterns too subtle for prompt artifacts alone.

> The durable asset is the learning loop, not the runtime shell.

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
│   └── codex/
│       ├── README.md
│       ├── context_builder.py
│       ├── item_mapper.py
│       └── sdk_runner.py
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
# 252 tests, all passing in <0.2s
# no network, no MLX, no Torch, no Ollama, no real model calls
```

The full suite imports cleanly without any optional ML dependency installed (MLX, Torch, Transformers, PEFT, TRL, Unsloth, requests, PyYAML are all lazy or test-injected).

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

### v1 — Symbolic substrate

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 0 — Architecture lock | Done | `spec.md`, `spec-validation.md`, `plan.md` |
| 1 — Core data plane | Done | `episode/`, `replay/` |
| 2 — Distillation | Done | `distill/skills.py`, `distill/dreams.py`, `distill/program.py` |
| 3 — Codex adapter | Done | `adapters/codex/` grounded in the local SDK |

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
