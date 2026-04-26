"""Tests for Ollama serving helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cl_layer.serve.modelfile import generate_modelfile, write_modelfile
from cl_layer.serve.ollama_create import ollama_create, ollama_exists
from cl_layer.serve.ollama_smoke import smoke_test_ollama


class TestModelfile:
    def test_generate_modelfile_contains_model_path(self):
        content = generate_modelfile("/tmp/model.gguf", "cl-agent-qwen:gen-1")
        assert "/tmp/model.gguf" in content
        assert "FROM /tmp/model.gguf" in content

    def test_generate_modelfile_contains_template(self):
        content = generate_modelfile("/tmp/model.gguf", "cl-agent-qwen:gen-1")
        assert "ollama" in content.lower() or "template" in content.lower()

    def test_generate_modelfile_custom_system(self):
        content = generate_modelfile("/tmp/model.gguf", "cl-agent-qwen:gen-1", system_prompt="Be concise.")
        assert "Be concise." in content

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
