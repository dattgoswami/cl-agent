"""Chat-template config + dataset rendering for SFT.

Single source of truth for the ChatML template the student model is trained on.
``serve/modelfile.py`` reads the same ``ChatTemplate`` and emits an Ollama
``TEMPLATE`` that produces byte-identical ChatML at inference time. Keeping
both sides on this one config is what prevents the train/serve template
mismatch the spec calls out as a primary risk.

No optional/heavy dependencies are imported here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cl_layer.dataset.example_schema import TrainingExample


@dataclass(frozen=True)
class ChatTemplate:
    """ChatML template configuration. Defaults match Qwen2.5-Coder-Instruct.

    Fields are *content* (system message text) and *role markers* (the literal
    ChatML boundary tokens). Together they define the on-the-wire string
    rendered for each turn.
    """

    system_prompt: str = "You are a helpful coding assistant."
    role_start: str = "<|im_start|>"
    role_end: str = "<|im_end|>"

    @property
    def eos_token(self) -> str:
        return self.role_end

    @property
    def stop_tokens(self) -> tuple[str, ...]:
        return (self.role_end, self.role_start)


DEFAULT_CHAT_TEMPLATE = ChatTemplate()


def example_to_messages(
    example: TrainingExample, template: ChatTemplate | None = None
) -> list[dict[str, str]]:
    """Build the canonical ``messages`` list for one training example."""
    template = template or DEFAULT_CHAT_TEMPLATE
    return [
        {"role": "system", "content": template.system_prompt},
        {"role": "user", "content": example.input_text},
        {"role": "assistant", "content": example.target_text},
    ]


def render_messages_chatml(
    messages: list[dict[str, str]], template: ChatTemplate | None = None
) -> str:
    """Render a list of ``{role, content}`` messages as a ChatML string.

    Each turn is wrapped exactly once. Turns are separated by a single
    newline; the trailing ``role_end`` is emitted by the turn itself, so the
    output never contains ``<|im_end|><|im_end|>``.
    """
    template = template or DEFAULT_CHAT_TEMPLATE
    return "\n".join(
        f"{template.role_start}{m['role']}\n{m['content']}{template.role_end}"
        for m in messages
    )


def render_example_chat(
    example: TrainingExample, template: ChatTemplate | None = None
) -> str:
    """Render one example as a ChatML string (system + user + assistant)."""
    return render_messages_chatml(example_to_messages(example, template), template)


def render_example_jsonl(
    example: TrainingExample, template: ChatTemplate | None = None
) -> str:
    """Render one example as a single JSONL line: ``{"messages": [...]}``."""
    record = {"messages": example_to_messages(example, template)}
    return json.dumps(record, ensure_ascii=False)


def render_examples_chatl(
    examples: list[TrainingExample], template: ChatTemplate | None = None
) -> list[str]:
    """Render a list of examples as JSON-Lines suitable for a dataset file.

    Each element is a JSON object string with a ``messages`` array containing
    ``system``/``user``/``assistant`` roles. Write the list to disk by
    joining with ``"\\n"`` to get a valid ``.jsonl`` file.
    """
    return [render_example_jsonl(ex, template) for ex in examples]
