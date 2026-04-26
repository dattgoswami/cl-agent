"""Tests for Ollama serving helpers."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cl_layer.dataset.render_chat import (
    ChatTemplate,
    DEFAULT_CHAT_TEMPLATE,
    render_messages_chatml,
)
from cl_layer.serve.modelfile import generate_modelfile, write_modelfile
from cl_layer.serve.ollama_create import ollama_create, ollama_exists
from cl_layer.serve.ollama_smoke import smoke_test_ollama


class TestModelfile:
    def test_generate_modelfile_contains_model_path(self):
        content = generate_modelfile("/tmp/model.gguf", "cl-agent-qwen:gen-1")
        assert "/tmp/model.gguf" in content
        assert "FROM /tmp/model.gguf" in content

    def test_system_block_has_opening_and_closing_triple_quotes(self):
        content = generate_modelfile("/tmp/model.gguf", system_prompt="hello world")
        # The whole SYSTEM block must be triple-quoted.
        match = re.search(r'^SYSTEM """(.*?)"""', content, re.MULTILINE | re.DOTALL)
        assert match is not None, f"no triple-quoted SYSTEM block in:\n{content}"
        assert match.group(1) == "hello world"

    def test_template_block_has_opening_and_closing_triple_quotes(self):
        content = generate_modelfile("/tmp/model.gguf")
        match = re.search(r'^TEMPLATE """(.*?)"""', content, re.MULTILINE | re.DOTALL)
        assert match is not None, f"no triple-quoted TEMPLATE block in:\n{content}"
        body = match.group(1)
        # Body must use Ollama Go-template variables, not Python format placeholders.
        assert "{{ .Prompt }}" in body
        assert "{{ .Response }}" in body
        assert "{{ .System }}" in body
        assert "{content}" not in body

    def test_modelfile_does_not_have_unbalanced_system_directive(self):
        # Catches the prior bug where SYSTEM was emitted as
        # `SYSTEM <prompt>` followed only by a *closing* triple-quote.
        content = generate_modelfile("/tmp/model.gguf", system_prompt="prompt")
        # No SYSTEM line that lacks an opening triple-quote.
        for line in content.splitlines():
            if line.startswith("SYSTEM"):
                assert line.startswith('SYSTEM """'), (
                    f"SYSTEM line missing opening triple-quote: {line!r}"
                )

    def test_modelfile_has_stop_parameters_for_chatml_boundaries(self):
        content = generate_modelfile("/tmp/model.gguf")
        assert 'PARAMETER stop "<|im_end|>"' in content
        assert 'PARAMETER stop "<|im_start|>"' in content

    def test_modelfile_has_temperature_and_ctx_parameters(self):
        content = generate_modelfile("/tmp/model.gguf", temperature=0.3, num_ctx=8192)
        assert "PARAMETER temperature 0.3" in content
        assert "PARAMETER num_ctx 8192" in content

    def test_modelfile_custom_system_prompt(self):
        content = generate_modelfile("/tmp/model.gguf", system_prompt="Be concise.")
        assert 'SYSTEM """Be concise."""' in content

    def test_modelfile_uses_default_system_when_unset(self):
        content = generate_modelfile("/tmp/model.gguf")
        assert f'SYSTEM """{DEFAULT_CHAT_TEMPLATE.system_prompt}"""' in content

    def test_modelfile_golden(self):
        """Pin the full Modelfile output for the default template so any drift
        between trainer and Ollama runtime is caught immediately."""
        tmpl = ChatTemplate(system_prompt="SYS")
        content = generate_modelfile(
            "/tmp/model.gguf",
            chat_template=tmpl,
            temperature=0.7,
            num_ctx=4096,
        )
        expected = (
            "FROM /tmp/model.gguf\n"
            "\n"
            "PARAMETER temperature 0.7\n"
            "PARAMETER num_ctx 4096\n"
            'PARAMETER stop "<|im_end|>"\n'
            'PARAMETER stop "<|im_start|>"\n'
            "\n"
            'SYSTEM """SYS"""\n'
            "\n"
            'TEMPLATE """{{ if .System }}<|im_start|>system\n'
            "{{ .System }}<|im_end|>\n"
            "{{ end }}{{ if .Prompt }}<|im_start|>user\n"
            "{{ .Prompt }}<|im_end|>\n"
            "{{ end }}<|im_start|>assistant\n"
            '{{ .Response }}<|im_end|>"""\n'
        )
        assert content == expected

    def test_modelfile_template_body_matches_trainer_chatml(self):
        """The TEMPLATE body, after substituting Ollama variables with concrete
        values, must equal what ``render_messages_chatml`` produces for the
        same system/user/assistant content. This is the core train↔serve
        parity guarantee."""
        tmpl = ChatTemplate(system_prompt="ignored")
        content = generate_modelfile("/tmp/model.gguf", chat_template=tmpl)
        match = re.search(r'^TEMPLATE """(.*?)"""', content, re.MULTILINE | re.DOTALL)
        assert match is not None
        body = match.group(1)

        # Simulate Ollama's Go-template substitution: both .System and .Prompt
        # are present, so the conditional branches both render.
        substituted = (
            body.replace("{{ if .System }}", "")
            .replace("{{ end }}{{ if .Prompt }}", "")
            .replace("{{ end }}", "")
            .replace("{{ .System }}", "S")
            .replace("{{ .Prompt }}", "U")
            .replace("{{ .Response }}", "A")
        )
        trainer_view = render_messages_chatml(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A"},
            ],
            tmpl,
        )
        assert substituted == trainer_view

    def test_modelfile_no_double_eos_after_substitution(self):
        content = generate_modelfile("/tmp/model.gguf")
        # The TEMPLATE body itself must not contain back-to-back EOS tokens.
        assert "<|im_end|><|im_end|>" not in content

    def test_write_modelfile(self, tmp_path: Path):
        content = "FROM /tmp/model.gguf\n"
        path = tmp_path / "Modelfile"
        write_modelfile(content, str(path))
        assert path.exists()
        assert path.read_text() == content


class TestOllamaCreate:
    def test_ollama_create_raises_on_missing_modelfile(self):
        with pytest.raises(FileNotFoundError):
            ollama_create("test-model", "/nonexistent/Modelfile")

    def test_ollama_create_calls_subprocess(self, tmp_path: Path):
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM /tmp/model.gguf\n")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""
        with patch("cl_layer.serve.ollama_create.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = ollama_create("cl-agent-qwen:gen-1", modelfile)
            assert result.returncode == 0
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "ollama" in call_args[0][0]
            assert "create" in call_args[0][0]

    def test_ollama_create_raises_on_failure(self, tmp_path: Path):
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM /tmp/model.gguf\n")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error: model not found"
        mock_result.stdout = ""
        with patch("cl_layer.serve.ollama_create.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            with pytest.raises(RuntimeError, match="ollama create failed"):
                ollama_create("cl-agent-qwen:gen-1", modelfile)

    def test_ollama_exists_true(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "cl-agent-qwen:gen-1\nollama/ollama:latest\n"
        with patch("cl_layer.serve.ollama_create.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            assert ollama_exists("cl-agent-qwen:gen-1") is True

    def test_ollama_exists_false(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ollama/ollama:latest\n"
        with patch("cl_layer.serve.ollama_create.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            assert ollama_exists("cl-agent-qwen:gen-1") is False

    def test_ollama_exists_when_ollama_missing(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("cl_layer.serve.ollama_create.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            assert ollama_exists("cl-agent-qwen:gen-1") is False


class TestOllamaSmoke:
    def test_smoke_all_pass(self):
        mock_client = MagicMock()
        mock_client.post.return_value = {"response": "Here is the code you requested."}
        result = smoke_test_ollama("test-model", ["prompt1", "prompt2"], client=mock_client)
        assert result["prompts_tested"] == 2
        assert result["prompts_passed"] == 2
        assert result["passed"] is True

    def test_smoke_with_empty_response(self):
        mock_client = MagicMock()
        mock_client.post.return_value = {"response": ""}
        result = smoke_test_ollama("test-model", ["prompt1"], client=mock_client)
        assert result["prompts_passed"] == 0
        assert result["passed"] is False
        assert len(result["errors"]) == 1

    def test_smoke_with_http_error(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("connection refused")
        result = smoke_test_ollama("test-model", ["prompt1"], client=mock_client)
        assert result["prompts_passed"] == 0
        assert len(result["errors"]) == 1
        assert "connection refused" in result["errors"][0]

    def test_smoke_without_requests_library(self):
        with patch("builtins.__import__") as mock_import:
            # Make the requests import fail
            def raise_import_error(name, *args, **kwargs):
                if name == "requests":
                    raise ImportError("no requests")
                # For other imports, use the real loader
                import builtins
                real_import = builtins.__import__
                return real_import(name, *args, **kwargs)

            mock_import.side_effect = raise_import_error
            result = smoke_test_ollama("test-model", ["prompt1"])
            assert result["prompts_tested"] == 1
            assert result["prompts_passed"] == 0
            assert "requests library not available" in result["errors"]

    def test_smoke_partial_pass(self):
        responses = [
            {"response": "Here is the code"},
            {"response": ""},
            {"response": "Some output"},
        ]

        class PartialClient:
            def __init__(self, responses):
                self._responses = responses
                self._i = 0

            def post(self, url, json):
                resp = self._responses[self._i]
                self._i += 1
                return resp

        result = smoke_test_ollama("test-model", ["a", "b", "c"], client=PartialClient(responses))
        assert result["prompts_tested"] == 3
        assert result["prompts_passed"] == 2
        assert result["passed"] is False
