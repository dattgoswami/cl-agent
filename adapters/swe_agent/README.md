# SWE-agent Adapter

Import-first adapter for full SWE-agent `.traj` JSON files.

This phase intentionally supports **full SWE-agent only**. Mini-SWE-agent is
not imported here because this repo does not yet have a concrete local
mini-SWE trace contract.

## Supported Format

The importer expects one `.traj` JSON object with these top-level keys:

```json
{
  "environment": "swe_main",
  "trajectory": [
    {
      "response": "...",
      "thought": "...",
      "action": "pytest -q",
      "observation": "1 passed",
      "state": {"working_dir": "/repo", "open_file": "n/a"},
      "execution_time": 1.2
    }
  ],
  "history": [{"role": "system", "content": "..."}],
  "info": {
    "exit_status": "submitted",
    "submission": "diff --git ...",
    "model_stats": {"tokens_sent": 100, "tokens_received": 20}
  }
}
```

Both older SWE-agent fixtures with stringified `state` and newer fixtures with
structured `state`, `execution_time`, and `message_type` are supported. The
adapter also preserves an adjacent config file when present
(`config.yaml`, `<instance>.config.yaml`, `args.yaml`, or
`run_batch.config.yaml`) as redacted compact metadata in the episode events.
The episode records config character count and path, but never embeds a config
excerpt in `episodes.jsonl`.

## Usage

```python
from adapters.swe_agent import import_trajectory

episode = import_trajectory(
    "trajectories/run/pydicom__pydicom-1458.traj",
    task_domain="swe_bench",
    mode="baseline",
)

print(episode.task_id)
print(episode.outcome.status)
print(episode.patch_hash)
```

To append directly:

```python
from adapters.swe_agent import append_trajectory_episode

append_trajectory_episode(
    "trajectories/run/pydicom__pydicom-1458.traj",
    "data/episodes.jsonl",
    task_domain="swe_bench",
)
```

## Modes

Trajectory import is observational. `baseline` and `integrated` label how the
run was produced; they do not alter saved evidence.

For live command preparation, `ContextBuilder` writes a small explicit
SWE-agent config overlay:

- Baseline: no CL artifacts, `agent.templates.demonstrations=[]`,
  `put_demos_in_history=False`, and no `strategy_template` key in the overlay.
- Integrated: reads `PROGRAM.md` and `SKILLS.md` and injects them through
  `agent.templates.strategy_template`, which SWE-agent appends via its prompt
  mechanism after the instance prompt.

The overlay is JSON-formatted YAML so it remains inspectable without adding a
PyYAML dependency to the CL layer.

## Mapping

| SWE-agent evidence | CL episode mapping |
| --- | --- |
| `history[*]` | `agent_message` with role, message type, compact text, action, and thought |
| `trajectory[*].thought/response/action` | `agent_message` trajectory-step event |
| `trajectory[*].action/observation` | `command_execution` with command, observation excerpt, state, timing, and inferred test evidence |
| `edit`, `create`, `rm`, `mv`, `cp` actions | `file_change` action evidence with paths when recoverable |
| `info.submission` | authoritative `patch_text`, `patch_hash`, and final `file_change` |
| submit observation diff | fallback `patch_text` when `info.submission` is absent |
| explicit `tests_passed`, `verification`, or `evaluation_result` | `evaluation_result` only when present in `info` |
| explicit `resolved` or `benchmark_result` | `benchmark_result` only when present in `info` |

`outcome.files_touched` is derived from the final patch when a patch exists.
Temporary reproduction files created and later removed are still visible as
action-level `file_change` events but are not attributed as final patch files.

`stdout_excerpt` is reserved for the latest test-like command observation when
one exists. The submit diff remains in `patch_text` / `patch_hash`, not
`stdout_excerpt`.

`reward` is always `None` at import time. Later evaluation can compute reward
from the same raw evidence.

## Live Runner Boundary

`SWEAgentRunner` is deliberately a command/config boundary:

```python
from adapters.swe_agent import SWEAgentRunner

runner = SWEAgentRunner(
    artifacts_dir="data",
    run_artifacts_dir="runs/swe-agent-captures",
    base_config="/path/to/SWE-agent/config/default.yaml",
    timeout_seconds=3600,
)

preview = runner.preview(
    "Fix the failing parser test.",
    task_id="parser-fix",
    mode="integrated",
    repo_path="/path/to/repo",
    model="gpt-4o",
)

print(preview.config_path)
print(preview.args)
```

Execution requires an injected `command_runner`. Tests should not run Docker,
download benchmarks, call model APIs, or invoke real SWE-bench tasks.
When using the subprocess boundary, `timeout_seconds` is passed to
`subprocess.run`; a timeout returns exit code `124` with timeout text in stderr.

## Benchmark Notes

SWE-agent `.traj` files capture task-oriented coding runs and the final
submission patch. SWE-bench pass/fail is usually produced by a separate
evaluation step, not by the trajectory itself. This adapter records
`benchmark_result` only when the `.traj` `info` object explicitly contains a
resolved/benchmark result. Otherwise it records the patch and any test-like
commands as evidence without claiming benchmark success.
