"""Chat JSONL rendering with configurable template."""

from __future__ import annotations

import json
from dataclasses import dataclass

from cl_layer.dataset.example_schema import TrainingExample


@dataclass
class ChatTemplate:
    """A Qwen/Ollama-compatible chat template."""

    system_prompt: str
    user_format: str
    assistant_format: str
    eos_token: str = "<|im_end|>"

    def render_turn(self, role: str, content: str) -> str:
        if role == "system":
            return self.system_prompt
        if role == "user":
            return self.user_format.format(content=content)
        if role == "assistant":
            return self.assistant_format.format(content=content)
        raise ValueError(f"Unknown role: {role}")


# Default Qwen/Ollama-compatible template
DEFAULT_CHAT_TEMPLATE = ChatTemplate(
    system_prompt="<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>",
    user_format="<|im_start|>user\n{content}<|im_end|>",
    assistant_format="<|im_start|>assistant\n{content}<|im_end|>",
)


def render_example_chat(example: TrainingExample, template: ChatTemplate | None = None) -> str:
    """Render a single example as chat JSONL."""
    template = template or DEFAULT_CHAT_TEMPLATE
    user_content = example.input_text
    assistant_content = example.target_text

    lines = [template.render_turn("system", "")]
    lines.append(template.render_turn("user", user_content))
    lines.append(template.render_turn("assistant", assistant_content))

    return template.eos_token.join(line for line in lines if line)


def render_examples_chatl(examples: list[TrainingExample], template: ChatTemplate | None = None) -> list[str]:
    """Render a list of examples as chat JSONL lines."""
    return [render_example_chat(ex, template) for ex in examples]
