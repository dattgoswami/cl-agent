"""Generate a valid Ollama Modelfile for a GGUF artifact.

The TEMPLATE block here MUST produce ChatML strings that are byte-identical
to what ``dataset/render_chat.py`` produces at training time. Both sides
share ``ChatTemplate`` so the role markers and system message stay in sync.

No top-level imports of optional/heavy dependencies.
"""

from __future__ import annotations

from cl_layer.dataset.render_chat import DEFAULT_CHAT_TEMPLATE, ChatTemplate


def _build_template_body(template: ChatTemplate) -> str:
    """Build the body of the Ollama TEMPLATE directive.

    Uses Ollama Go-template variables ``{{ .System }}`` / ``{{ .Prompt }}`` /
    ``{{ .Response }}`` — never Python ``{content}`` placeholders.

    The expanded output (with all three present) matches
    ``render_messages_chatml`` for system + user + assistant.
    """
    rs, re_ = template.role_start, template.role_end
    return (
        f"{{{{ if .System }}}}{rs}system\n{{{{ .System }}}}{re_}\n{{{{ end }}}}"
        f"{{{{ if .Prompt }}}}{rs}user\n{{{{ .Prompt }}}}{re_}\n{{{{ end }}}}"
        f"{rs}assistant\n{{{{ .Response }}}}{re_}"
    )


def generate_modelfile(
    model_path: str,
    model_name: str | None = None,
    system_prompt: str | None = None,
    chat_template: ChatTemplate | None = None,
    temperature: float = 0.7,
    num_ctx: int = 4096,
) -> str:
    """Return a valid Ollama Modelfile string for a GGUF model.

    Guarantees:
    - ``FROM <model_path>`` line
    - ``PARAMETER`` lines for ``temperature``, ``num_ctx``, and ``stop`` for
      both ChatML boundary tokens
    - ``SYSTEM \"\"\"...\"\"\"`` block with both opening and closing triple
      quotes
    - ``TEMPLATE \"\"\"...\"\"\"`` block with both opening and closing triple
      quotes, using Ollama Go-template variables
    """
    tmpl = chat_template or DEFAULT_CHAT_TEMPLATE
    system = system_prompt if system_prompt is not None else tmpl.system_prompt
    template_body = _build_template_body(tmpl)

    lines: list[str] = []
    if model_name:
        lines.append(f"# Model: {model_name}")
    lines.extend(
        [
            f"FROM {model_path}",
            "",
            f"PARAMETER temperature {temperature}",
            f"PARAMETER num_ctx {num_ctx}",
            f'PARAMETER stop "{tmpl.role_end}"',
            f'PARAMETER stop "{tmpl.role_start}"',
            "",
            f'SYSTEM """{system}"""',
            "",
            f'TEMPLATE """{template_body}"""',
            "",
        ]
    )
    return "\n".join(lines)


def write_modelfile(content: str, path: str) -> None:
    """Write Modelfile content to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
