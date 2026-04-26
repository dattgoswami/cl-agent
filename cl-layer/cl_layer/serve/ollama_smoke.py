"""Smoke-test helper for Ollama models."""

from __future__ import annotations

from typing import Protocol


class HttpClient(Protocol):
    """Injectable HTTP client for smoke tests."""

    def post(self, url: str, json: dict) -> dict:
        ...


def smoke_test_ollama(
    model_name: str,
    prompts: list[str],
    client: HttpClient | None = None,
    base_url: str = "http://localhost:11434",
) -> dict:
    """Smoke-test an Ollama model against a set of prompts.

    Uses an injectable HttpClient (default: lazy-imported requests).
    Returns pass/fail counts and any errors.
    """
    if client is None:
        try:
            import requests

            client = _RequestsClient(f"{base_url}/api/generate")
        except ImportError:
            return {
                "model": model_name,
                "prompts_tested": len(prompts),
                "prompts_passed": 0,
                "errors": ["requests library not available"],
                "passed": False,
            }

    passed = 0
    errors: list[str] = []
    for prompt in prompts:
        try:
            response = client.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
            )
            text = response.get("response", "")
            if text and len(text.strip()) > 0:
                passed += 1
            else:
                errors.append(f"Empty response for prompt: {prompt[:50]}...")
        except Exception as e:
            errors.append(f"Error for prompt {prompt[:50]}: {e}")

    return {
        "model": model_name,
        "prompts_tested": len(prompts),
        "prompts_passed": passed,
        "errors": errors,
        "passed": passed == len(prompts),
    }


class _RequestsClient:
    def __init__(self, url: str) -> None:
        self._url = url

    def post(self, url: str, json: dict) -> dict:
        import requests

        return requests.post(url, json=json).json()
