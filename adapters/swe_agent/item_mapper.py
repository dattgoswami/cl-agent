from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Any

from cl_layer.episode.schema import EpisodeEvent, EpisodeOutcome

MAX_TEXT_CHARS = 4000
MAX_EXCERPT_CHARS = 2000
MAX_ACTION_CHARS = 1200

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)\s*$")
_DIFF_PLUS_RE = re.compile(r"^\+\+\+ b/(.*?)\s*$")
_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|unittest|npm\s+(run\s+)?test|pnpm\s+test|yarn\s+test|"
    r"vitest|go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|"
    r"ruff|flake8|mypy|eslint|tsc)\b",
    re.IGNORECASE,
)
_EXIT_CODE_RE = re.compile(
    r"(?:exit code|exited with code|returncode|return code)\s*:?\s*(-?\d+)",
    re.IGNORECASE,
)
_FAILED_TEST_RE = re.compile(
    r"\b(failed|failures?|errors?|traceback|assertionerror)\b",
    re.IGNORECASE,
)
_PASSED_TEST_RE = re.compile(r"\b(passed|success|successful|ok)\b", re.IGNORECASE)
_SIMPLE_YAML_KEY_RE = re.compile(r"^(?P<indent>[ \t]*)(?:-\s*)?(?P<key>[A-Za-z_][\w-]*)\s*:\s*(?P<value>.*)$")

_SUCCESS_EXIT_STATUSES = {
    "submitted",
    "success",
    "succeeded",
    "completed",
    "complete",
    "done",
}
_FAILURE_EXIT_PREFIXES = (
    "exit_",
    "failed",
    "failure",
    "error",
    "cancelled",
    "canceled",
    "timeout",
)


@dataclass(frozen=True)
class SWEAgentMappingResult:
    events: list[EpisodeEvent]
    outcome: EpisodeOutcome
    patch_text: str | None
    patch_hash: str | None
    tool_trace: list[dict] | None
    test_trace: list[dict] | None
    stdout_excerpt: str | None
    stderr_excerpt: str | None
    cost_tokens_prompt: int | None
    cost_tokens_completion: int | None
    latency_ms: float | None
    repo_path: str | None
    base_model_id: str | None
    verification_steps: list[dict] | None
    verification_score: float | None
    verification_failures: list[str] | None


def _truncate(value: object, limit: int = MAX_TEXT_CHARS) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...[truncated {len(value) - limit} chars]"


def _hash_patch(patch_text: str | None) -> str | None:
    if not patch_text:
        return None
    return f"sha256:{hashlib.sha256(patch_text.encode('utf-8')).hexdigest()}"


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _parse_json_string(value: object) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _parse_state(state: object) -> dict:
    parsed = _parse_json_string(state)
    return parsed if isinstance(parsed, dict) else {}


def _compact_state(state: object) -> dict:
    parsed = _parse_state(state)
    return {
        "working_dir": parsed.get("working_dir"),
        "open_file": parsed.get("open_file"),
    }


def _command_name(action: str | None) -> str | None:
    if not action:
        return None
    stripped = action.strip()
    if not stripped:
        return None
    return stripped.split(None, 1)[0]


def _extract_exit_code(observation: str | None) -> int | None:
    if not observation:
        return None
    match = _EXIT_CODE_RE.search(observation)
    return int(match.group(1)) if match else None


def _is_test_command(action: str | None) -> bool:
    return bool(action and _TEST_COMMAND_RE.search(action))


def _infer_test_passed(observation: str | None, exit_code: int | None) -> bool | None:
    if exit_code is not None:
        return exit_code == 0
    if not observation:
        return None
    if _FAILED_TEST_RE.search(observation):
        return False
    if _PASSED_TEST_RE.search(observation):
        return True
    return None


def paths_from_unified_diff(patch_text: str | None) -> list[str]:
    if not patch_text:
        return []
    paths: list[str] = []
    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\r")
        match = _DIFF_GIT_RE.match(line)
        if match:
            after = match.group(2)
            if after != "/dev/null":
                paths.append(after)
            continue
        match = _DIFF_PLUS_RE.match(line)
        if match:
            path = match.group(1)
            if path != "/dev/null":
                paths.append(path)
    return sorted(dict.fromkeys(path.strip() for path in paths if path.strip()))


def _strip_repo_prefix(path: str, repo_path: str | None) -> str:
    if not path:
        return path
    if repo_path and path.startswith(repo_path.rstrip("/") + "/"):
        return path[len(repo_path.rstrip("/")) + 1 :]
    return path


def _clamp_working_dir_to_repo_root(working_dir: str) -> str | None:
    text = working_dir.strip()
    if not text or text == "n/a":
        return None
    path = PurePosixPath(text)
    parts = [part for part in path.parts if part != "/"]
    if not parts:
        return "/" if text.startswith("/") else None
    root = parts[0]
    return f"/{root}" if path.is_absolute() else root


def _clean_path(path: object, repo_path: str | None) -> str | None:
    if not isinstance(path, str) or not path.strip() or path.strip() == "n/a":
        return None
    text = path.strip().strip("\"'")
    text = _strip_repo_prefix(text, repo_path)
    return text.lstrip("/") if text.startswith("/") else text


def _action_paths(action: str | None, state: object, repo_path: str | None) -> list[str]:
    if not action:
        return []
    command = _command_name(action)
    state_dict = _parse_state(state)
    stripped = action.strip()
    paths: list[str] = []

    if command in {"edit"}:
        path = _clean_path(state_dict.get("open_file"), repo_path)
        if path:
            paths.append(path)
    elif command in {"create", "open", "search_file"}:
        parts = stripped.split(maxsplit=2)
        if len(parts) >= 2:
            path = _clean_path(parts[1], repo_path)
            if path:
                paths.append(path)
    elif command in {"rm", "mv", "cp"}:
        parts = stripped.split()
        for part in parts[1:]:
            if part.startswith("-"):
                continue
            path = _clean_path(part, repo_path)
            if path:
                paths.append(path)

    return sorted(dict.fromkeys(paths))


def _extract_patch_from_info(info: dict) -> str | None:
    for key in ("submission", "patch", "model_patch", "diff"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_patch_from_submit_step(trajectory: list[dict]) -> str | None:
    for step in reversed(trajectory):
        action = step.get("action")
        observation = step.get("observation")
        if (
            isinstance(action, str)
            and action.strip().startswith("submit")
            and isinstance(observation, str)
            and "diff --git" in observation
        ):
            return observation
    return None


def _is_submitted_variant(normalized_exit_status: str) -> bool:
    return normalized_exit_status.startswith("submitted") and normalized_exit_status not in _SUCCESS_EXIT_STATUSES


def _extract_repo_path(raw: dict, trajectory: list[dict], info: dict) -> str | None:
    for value in (
        info.get("repo_path"),
        info.get("repo"),
        raw.get("repo_path"),
        raw.get("repo"),
    ):
        if isinstance(value, str) and value:
            return value
    for step in reversed(trajectory):
        state = _parse_state(step.get("state"))
        working_dir = state.get("working_dir")
        if isinstance(working_dir, str) and working_dir and working_dir != "n/a":
            return _clamp_working_dir_to_repo_root(working_dir)
    return None


def _model_name_from_mapping(config: dict) -> str | None:
    agent = config.get("agent")
    if isinstance(agent, dict):
        model = agent.get("model")
        if isinstance(model, dict) and isinstance(model.get("name"), str):
            return model["name"]
    model = config.get("model")
    if isinstance(model, dict) and isinstance(model.get("name"), str):
        return model["name"]
    return None


def _strip_yaml_scalar(value: str) -> str | None:
    text = value.split("#", 1)[0].strip()
    if not text:
        return None
    if text[0] in {"'", '"'} and text[-1:] == text[0]:
        text = text[1:-1]
    return text.strip() or None


def _model_from_yaml_text(config: str) -> str | None:
    stack: list[tuple[int, str]] = []
    for raw_line in config.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        match = _SIMPLE_YAML_KEY_RE.match(raw_line)
        if not match:
            continue
        indent = len(match.group("indent").expandtabs(2))
        key = match.group("key")
        value = match.group("value")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent_path = [item[1] for item in stack]
        scalar = _strip_yaml_scalar(value)
        if key == "name" and scalar and parent_path in (["agent", "model"], ["model"]):
            return scalar
        stack.append((indent, key))
    return None


def _model_from_config(config: object) -> str | None:
    if isinstance(config, dict):
        return _model_name_from_mapping(config)
    if isinstance(config, str):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            parsed = None
        else:
            try:
                loaded = yaml.safe_load(config)
            except Exception:
                parsed = None
            else:
                parsed = loaded if isinstance(loaded, dict) else None
        if isinstance(parsed, dict):
            model = _model_name_from_mapping(parsed)
            if model:
                return model
        return _model_from_yaml_text(config)
    return None


def _base_model_id(raw: dict, info: dict, source_config: object | None, source_path: str | None) -> str | None:
    for value in (
        info.get("model"),
        info.get("model_name"),
        info.get("model_name_or_path"),
        raw.get("model"),
    ):
        if isinstance(value, str) and value:
            return value

    replay_config = raw.get("replay_config")
    parsed_replay = _parse_json_string(replay_config)
    model = _model_from_config(parsed_replay)
    if model:
        return model
    model = _model_from_config(source_config)
    if model:
        return model

    if source_path:
        run_name = PurePosixPath(source_path).parent.name
        if "__" in run_name:
            return run_name.split("__", 1)[0]
    return None


def _explicit_bool(info: dict, *keys: str) -> bool | None:
    for key in keys:
        value = info.get(key)
        if isinstance(value, bool):
            return value
    return None


def _explicit_summary(info: dict) -> str | None:
    for key in (
        "verification_summary",
        "evaluation_summary",
        "benchmark_summary",
        "test_summary",
        "summary",
    ):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _explicit_verification_steps(info: dict) -> list[dict] | None:
    value = info.get("verification_steps") or info.get("evaluation_steps")
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return None


def _explicit_verification_score(info: dict) -> float | None:
    value = info.get("verification_score") or info.get("evaluation_score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _explicit_failures(info: dict) -> list[str] | None:
    value = info.get("verification_failures") or info.get("failures")
    if isinstance(value, list):
        failures = [str(item) for item in value if item]
        return failures or None
    return None


def _model_stats(info: dict) -> dict:
    value = info.get("model_stats")
    return value if isinstance(value, dict) else {}


def _int_from_stats(stats: dict, *keys: str) -> int | None:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _total_execution_time_ms(trajectory: list[dict]) -> float | None:
    total = 0.0
    seen = False
    for step in trajectory:
        value = step.get("execution_time")
        if isinstance(value, (int, float)):
            total += float(value)
            seen = True
    return total * 1000 if seen else None


def _final_response(trajectory: list[dict], history: list[dict]) -> str | None:
    for step in reversed(trajectory):
        response = step.get("response") or step.get("thought")
        if isinstance(response, str) and response.strip():
            return _truncate(response)
    for item in reversed(history):
        if item.get("role") == "assistant" and isinstance(item.get("content"), str):
            return _truncate(item["content"])
    return None


def _add_metadata_event(
    *,
    events: list[EpisodeEvent],
    timestamp: datetime,
    raw: dict,
    info: dict,
    source_config: object | None,
    source_config_path: str | None,
    source_path: str | None,
    base_model_id: str | None,
    repo_path: str | None,
) -> None:
    config_payload: dict[str, object] | None = None
    if isinstance(source_config, dict):
        config_payload = {"type": "dict", "keys": sorted(str(key) for key in source_config.keys())}
    elif isinstance(source_config, str) and source_config:
        config_payload = {
            "type": "text",
            "chars": len(source_config),
            "path": source_config_path,
            "redacted": True,
        }

    events.append(
        EpisodeEvent(
            kind="agent_message",
            timestamp=timestamp,
            payload={
                "source": "swe_agent_trajectory",
                "swe_agent_event_type": "run_metadata",
                "environment": raw.get("environment"),
                "exit_status": info.get("exit_status"),
                "base_model_id": base_model_id,
                "repo_path": repo_path,
                "source_path": source_path,
                "model_stats": _model_stats(info) or None,
                "config": config_payload,
            },
        )
    )


def _add_history_events(events: list[EpisodeEvent], history: list[dict], timestamp: datetime) -> None:
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        action = item.get("action")
        thought = item.get("thought")
        events.append(
            EpisodeEvent(
                kind="agent_message",
                timestamp=timestamp,
                payload={
                    "source": "swe_agent_history",
                    "role": item.get("role"),
                    "agent": item.get("agent"),
                    "message_type": item.get("message_type"),
                    "text": _truncate(content),
                    "action": _truncate(action, MAX_ACTION_CHARS),
                    "thought": _truncate(thought, MAX_ACTION_CHARS),
                    "tool_call_ids": item.get("tool_call_ids"),
                    "history_index": index,
                },
            )
        )


def _add_explicit_result_events(
    events: list[EpisodeEvent],
    info: dict,
    timestamp: datetime,
) -> None:
    for key in ("evaluation_result", "verification", "verifier"):
        value = info.get(key)
        if isinstance(value, dict):
            events.append(
                EpisodeEvent(
                    kind="evaluation_result",
                    timestamp=timestamp,
                    payload={"source": "swe_agent_info", "key": key, "result": value},
                )
            )

    tests_passed = _explicit_bool(info, "tests_passed", "verification_passed")
    if tests_passed is not None:
        events.append(
            EpisodeEvent(
                kind="evaluation_result",
                timestamp=timestamp,
                payload={"source": "swe_agent_info", "tests_passed": tests_passed},
            )
        )

    for key in ("benchmark_result", "swe_bench_result"):
        value = info.get(key)
        if isinstance(value, dict):
            events.append(
                EpisodeEvent(
                    kind="benchmark_result",
                    timestamp=timestamp,
                    payload={"source": "swe_agent_info", "key": key, "result": value},
                )
            )

    resolved = _explicit_bool(info, "resolved", "swe_bench_resolved")
    if resolved is not None:
        events.append(
            EpisodeEvent(
                kind="benchmark_result",
                timestamp=timestamp,
                payload={"source": "swe_agent_info", "resolved": resolved},
            )
        )


def _status_from_evidence(
    *,
    exit_status: object,
    patch_text: str | None,
    tests_passed: bool | None,
    events: list[EpisodeEvent],
    info: dict,
) -> tuple[str, str | None]:
    if not events:
        return "failed", "no parseable SWE-agent trajectory events"

    if isinstance(exit_status, int):
        if exit_status == 0:
            return "completed", None
        return ("partial" if patch_text else "failed"), f"SWE-agent exit_status={exit_status}"

    if isinstance(exit_status, str):
        normalized = exit_status.strip().lower()
    elif exit_status is None:
        normalized = None
    else:
        normalized = str(exit_status).strip().lower()

    explicit_error = info.get("error")
    if explicit_error:
        return ("partial" if patch_text else "failed"), str(explicit_error)

    if normalized in _SUCCESS_EXIT_STATUSES:
        return "completed", None

    if tests_passed is False:
        return ("partial" if patch_text else "failed"), "latest test-like command failed"

    if normalized:
        if _is_submitted_variant(normalized):
            return ("partial" if patch_text else "failed"), f"SWE-agent exit_status={exit_status}"
        if normalized.startswith(_FAILURE_EXIT_PREFIXES):
            return ("partial" if patch_text else "failed"), f"SWE-agent exit_status={exit_status}"
        if patch_text:
            return "partial", f"SWE-agent exit_status={exit_status}"
        return "failed", f"SWE-agent exit_status={exit_status}"

    if patch_text:
        return "completed", None
    return "completed", None


def _step_timestamps(trajectory: list[dict], started_at: datetime) -> tuple[list[datetime], datetime]:
    timestamps: list[datetime] = []
    cumulative = 0.0
    saw_execution_time = False
    for step in trajectory:
        execution_time = step.get("execution_time")
        if isinstance(execution_time, (int, float)) and execution_time >= 0:
            cumulative += float(execution_time)
            saw_execution_time = True
        timestamps.append(started_at + timedelta(seconds=cumulative))
    fallback_end = started_at + timedelta(seconds=cumulative) if saw_execution_time else started_at
    return timestamps, fallback_end


def map_swe_agent_trajectory(
    raw: dict,
    *,
    timestamp: datetime,
    ended_at: datetime | None = None,
    source_path: str | None = None,
    source_config: object | None = None,
    source_config_path: str | None = None,
) -> SWEAgentMappingResult:
    trajectory = raw.get("trajectory") if isinstance(raw.get("trajectory"), list) else []
    history = raw.get("history") if isinstance(raw.get("history"), list) else []
    info = _as_dict(raw.get("info"))

    trajectory_steps = [step for step in trajectory if isinstance(step, dict)]
    history_items = [item for item in history if isinstance(item, dict)]
    step_timestamps, derived_ended_at = _step_timestamps(trajectory_steps, timestamp)
    summary_timestamp = ended_at or derived_ended_at
    if ended_at is not None:
        step_timestamps = [min(step_timestamp, ended_at) for step_timestamp in step_timestamps]

    patch_text = _extract_patch_from_info(info) or _extract_patch_from_submit_step(trajectory_steps)
    patch_hash = _hash_patch(patch_text)
    patch_files = paths_from_unified_diff(patch_text)
    repo_path = _extract_repo_path(raw, trajectory_steps, info)
    base_model_id = _base_model_id(raw, info, source_config, source_path)
    stats = _model_stats(info)

    events: list[EpisodeEvent] = []
    tool_trace: list[dict] = []
    test_trace: list[dict] = []
    mutating_action_files: set[str] = set()
    test_outputs: list[str] = []
    failure_outputs: list[str] = []
    latest_test_passed: bool | None = None

    _add_metadata_event(
        events=events,
        timestamp=timestamp,
        raw=raw,
        info=info,
        source_config=source_config,
        source_config_path=source_config_path,
        source_path=source_path,
        base_model_id=base_model_id,
        repo_path=repo_path,
    )
    _add_history_events(events, history_items, timestamp)

    for index, step in enumerate(trajectory_steps):
        step_timestamp = step_timestamps[index]
        action = step.get("action") if isinstance(step.get("action"), str) else ""
        observation = step.get("observation") if isinstance(step.get("observation"), str) else ""
        command = action.strip() or None
        command_name = _command_name(command)
        exit_code = _extract_exit_code(observation)
        execution_time = step.get("execution_time")
        state = _compact_state(step.get("state"))
        is_test = _is_test_command(command)
        test_passed = _infer_test_passed(observation, exit_code) if is_test else None
        if test_passed is not None:
            latest_test_passed = test_passed

        if observation:
            if is_test:
                test_outputs.append(observation)
            if (exit_code is not None and exit_code != 0) or test_passed is False:
                failure_outputs.append(observation)

        events.append(
            EpisodeEvent(
                kind="agent_message",
                timestamp=step_timestamp,
                payload={
                    "source": "swe_agent_trajectory",
                    "swe_agent_event_type": "trajectory_step",
                    "step_index": index,
                    "thought": _truncate(step.get("thought"), MAX_ACTION_CHARS),
                    "response": _truncate(step.get("response"), MAX_TEXT_CHARS),
                    "action": _truncate(command, MAX_ACTION_CHARS),
                    "state": state,
                },
            )
        )

        if command:
            command_payload = {
                "source": "swe_agent_trajectory",
                "step_index": index,
                "command": command,
                "command_name": command_name,
                "exit_code": exit_code,
                "execution_time_seconds": execution_time if isinstance(execution_time, (int, float)) else None,
                "state": state,
                "observation": _truncate(observation, MAX_TEXT_CHARS),
                "is_test_command": is_test,
                "test_passed": test_passed,
            }
            events.append(EpisodeEvent(kind="command_execution", timestamp=step_timestamp, payload=command_payload))
            tool_trace.append(command_payload)

            if is_test:
                test_record = {
                    "command": command,
                    "exit_code": exit_code,
                    "passed": test_passed,
                    "source": "swe_agent_trajectory",
                    "step_index": index,
                }
                test_trace.append(test_record)

        action_paths = _action_paths(command, step.get("state"), repo_path)
        mutating = command_name in {"edit", "create", "rm", "mv", "cp"}
        if mutating and action_paths:
            mutating_action_files.update(action_paths)
            events.append(
                EpisodeEvent(
                    kind="file_change",
                    timestamp=step_timestamp,
                    payload={
                        "source": "swe_agent_action",
                        "step_index": index,
                        "operation": command_name,
                        "paths": action_paths,
                        "patch_hash": None,
                        "patch_chars": 0,
                        "action_excerpt": _truncate(command, MAX_ACTION_CHARS),
                        "observation": _truncate(observation, MAX_EXCERPT_CHARS),
                    },
                )
            )

    if patch_text or patch_files:
        events.append(
            EpisodeEvent(
                kind="file_change",
                timestamp=summary_timestamp,
                payload={
                    "source": "swe_agent_submission",
                    "operation": "submission",
                    "paths": patch_files,
                    "patch_hash": patch_hash,
                    "patch_chars": len(patch_text) if patch_text else 0,
                    "patch_excerpt": _truncate(patch_text, MAX_EXCERPT_CHARS),
                    "exit_status": info.get("exit_status"),
                },
            )
        )

    _add_explicit_result_events(events, info, summary_timestamp)

    explicit_tests_passed = _explicit_bool(info, "tests_passed", "verification_passed", "resolved")
    tests_passed = explicit_tests_passed if explicit_tests_passed is not None else latest_test_passed

    explicit_summary = _explicit_summary(info)
    if explicit_summary:
        verification_summary = explicit_summary
    elif tests_passed is not None and test_trace:
        verification_summary = "Latest test-like command passed." if tests_passed else "Latest test-like command failed."
    elif _explicit_bool(info, "resolved", "swe_bench_resolved") is not None:
        verification_summary = "Benchmark marked resolved." if tests_passed else "Benchmark marked unresolved."
    else:
        verification_summary = None

    status, escalation_reason = _status_from_evidence(
        exit_status=info.get("exit_status"),
        patch_text=patch_text,
        tests_passed=tests_passed,
        events=events,
        info=info,
    )

    outcome_files = patch_files if patch_files else sorted(mutating_action_files)

    return SWEAgentMappingResult(
        events=events,
        outcome=EpisodeOutcome(
            status=status,  # type: ignore[arg-type]
            tests_passed=tests_passed,
            verification_summary=verification_summary,
            escalation_reason=escalation_reason,
            files_touched=outcome_files,
            final_response=_final_response(trajectory_steps, history_items),
        ),
        patch_text=patch_text,
        patch_hash=patch_hash,
        tool_trace=tool_trace or None,
        test_trace=test_trace or None,
        stdout_excerpt=_truncate(test_outputs[-1], MAX_EXCERPT_CHARS) if test_outputs else None,
        stderr_excerpt=_truncate(failure_outputs[-1], MAX_EXCERPT_CHARS) if failure_outputs else None,
        cost_tokens_prompt=_int_from_stats(stats, "tokens_sent", "prompt_tokens", "input_tokens"),
        cost_tokens_completion=_int_from_stats(stats, "tokens_received", "completion_tokens", "output_tokens"),
        latency_ms=_total_execution_time_ms(trajectory_steps),
        repo_path=repo_path,
        base_model_id=base_model_id,
        verification_steps=_explicit_verification_steps(info),
        verification_score=_explicit_verification_score(info),
        verification_failures=_explicit_failures(info),
    )
