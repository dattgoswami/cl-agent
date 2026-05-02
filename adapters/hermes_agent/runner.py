from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

from .context_builder import ContextBuilder, HermesRunContext


@dataclass(frozen=True)
class HermesRunResult:
    context: HermesRunContext
    result: dict


ConversationRunner = Callable[[HermesRunContext, Mapping[str, object]], dict]


class HermesAgentRunner:
    """
    Thin, injectable live runner boundary for Hermes.

    The first Hermes adapter implementation is import-first.  This runner only
    builds a mode-aware context and delegates execution to an injected callable,
    so tests do not require model keys, network access, Hermes gateways, or
    background services.
    """

    def __init__(
        self,
        *,
        artifacts_dir: str | Path | None = None,
        conversation_runner: ConversationRunner | None = None,
    ) -> None:
        self.context_builder = ContextBuilder(artifacts_dir)
        self.conversation_runner = conversation_runner or self._unconfigured_runner

    def run(
        self,
        task_prompt: str,
        *,
        mode: Literal["baseline", "integrated"] = "baseline",
        cwd: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
        enabled_toolsets: list[str] | None = None,
        extra_agent_kwargs: Mapping[str, object] | None = None,
    ) -> HermesRunResult:
        context = self.context_builder.build(task_prompt, mode=mode, cwd=cwd)
        kwargs: dict[str, object] = context.agent_kwargs()
        if cwd is not None:
            kwargs["cwd"] = cwd
        if model is not None:
            kwargs["model"] = model
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations
        if enabled_toolsets is not None:
            kwargs["enabled_toolsets"] = enabled_toolsets
        kwargs.update(dict(extra_agent_kwargs or {}))

        result = self.conversation_runner(context, kwargs)
        return HermesRunResult(context=context, result=result)

    @staticmethod
    def _unconfigured_runner(_context: HermesRunContext, _kwargs: Mapping[str, object]) -> dict:
        raise RuntimeError(
            "HermesAgentRunner requires an injected conversation_runner. "
            "Use trajectory import for deterministic CL capture."
        )
