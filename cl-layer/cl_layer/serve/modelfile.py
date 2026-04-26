"""Generate Ollama Modelfile content for GGUF artifacts."""

from __future__ import annotations

from cl_layer.dataset.render_chat import ChatTemplate, DEFAULT_CHAT_TEMPLATE


def generate_modelfile(
    model_path: str,
    model_name: str,
    system_prompt: str | None = None,
    chat_template: ChatTemplate | None = None,
) -> str:
    """Generate an Ollama Modelfile string for a GGUF model."""
    chat_template = chat_template or DEFAULT_CHAT_TEMPLATE
    system = system_prompt or chat_template.system_prompt or "You are a helpful coding assistant."

    lines = [
        f"FROM {model_path}",
        "",
        f"PARAMETER temperature 0.7",
        f"PARAMETER num_ctx 4096",
        "",
        "SYSTEM """ + system + '"""',
        "",
        "# Chat template matching training template",
        f"TEMPLATE {chat_template.user_format} {{ .Prompt }} {chat_template.assistant_format} {{ .Response }}{chat_template.eos_token}",
    ]
    return "\n".join(lines)


def write_modelfile(content: str, path: str) -> None:
    """Write Modelfile content to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
