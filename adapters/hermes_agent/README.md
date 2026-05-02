# Hermes Agent Adapter

Import-first adapter for Hermes Agent batch trajectories.

## Trace Source

The supported input is Hermes' ShareGPT-style trajectory JSONL written by
`batch_runner.py`:

```json
{"prompt_index":0,"conversations":[{"from":"system","value":"..."},{"from":"human","value":"..."},{"from":"gpt","value":"<tool_call>...</tool_call>"},{"from":"tool","value":"<tool_response>...</tool_response>"}],"metadata":{"batch_num":0,"timestamp":"2026-04-20T10:00:00","model":"..."},"completed":true,"partial":false}
```

This is the first supported source because Hermes batch runs already set
`skip_memory=True` and `skip_context_files=True`, which avoids contaminating CL
attribution with Hermes' native learning loop. Live CLI/TUI scraping is not
supported. SQLite/session-log import is a follow-up.

## Usage

```python
from adapters.hermes_agent import import_trajectory_jsonl

episodes = import_trajectory_jsonl(
    "runs/my_run/trajectories.jsonl",
    task_id_prefix="hermes-task",
    task_domain="python",
    mode="baseline",
)
```

To append directly:

```python
from adapters.hermes_agent import append_trajectory_episodes

append_trajectory_episodes(
    "runs/my_run/trajectories.jsonl",
    "data/episodes/hermes.jsonl",
    task_id_prefix="hermes-task",
    task_domain="python",
)
```

## Modes

Baseline mode is for controlled experiments. The live runner context sets:

- `skip_memory=True`
- `skip_context_files=True`
- `persist_session=False`
- no CL artifact injection

Integrated mode injects `PROGRAM.md` and `SKILLS.md` through Hermes'
`ephemeral_system_prompt` constructor argument and leaves Hermes native memory,
context files, and session persistence enabled. That is useful for daily use,
but CL attribution should treat Hermes memory/skills as native-agent evidence,
not substrate-owned fields.

## Mapping

The importer maps:

- `terminal`, `bash`, `shell`, `execute_code` -> `command_execution`
- `read_file`, `write_file`, `patch`, `search_files` -> `file_change`
- `mcp_*` tools -> `mcp_tool_call`
- user, assistant, system, memory, skill, session-search, and subagent evidence -> `agent_message`

Memory and skill events stay in event payloads with discriminators such as
`hermes_event_type="memory_event"` and `hermes_event_type="skill_event"`.
The adapter does not add schema fields.

For test-like terminal commands, `test_trace` keeps every observed command but
`outcome.tests_passed` and `verification_summary` summarize the latest one.

## Limitations

- No live Hermes process is started by default. `HermesAgentRunner` requires an
  injected callable.
- No Hermes SQLite/session-log importer yet.
- Batch trajectories do not contain per-message timestamps, so events in one
  trajectory share the entry timestamp.
- Tool-call pairing follows Hermes' saved response names and call order.
- MCP tool names are parsed best-effort from Hermes' flattened
  `mcp_<server>_<tool>` name. If an MCP server name itself contains underscores,
  the parsed `server` / `tool` split may be ambiguous; the raw
  `hermes_tool_name` is always preserved.
- Bare `read` and `write` are treated as file-operation aliases for compatibility
  with adjacent coding-agent traces. A future Hermes tool that uses those names
  for non-filesystem work would need an adapter mapping update.
