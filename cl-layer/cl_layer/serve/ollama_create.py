"""Run `ollama create` via subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path


def ollama_create(
    model_name: str,
    modelfile_path: str | Path,
) -> subprocess.CompletedProcess:
    """Create an Ollama model from a Modelfile.

    Uses `ollama create` with shell=False for safety.
    """
    modelfile_path = Path(modelfile_path)
    if not modelfile_path.exists():
        raise FileNotFoundError(f"Modelfile not found: {modelfile_path}")

    result = subprocess.run(
        ["ollama", "create", model_name, "--file", str(modelfile_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed: {result.stderr}")
    return result


def ollama_exists(model_name: str) -> bool:
    """Check if an Ollama model is already present."""
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return False
    return model_name in result.stdout
