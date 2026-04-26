"""Model export utilities."""

from __future__ import annotations

from pathlib import Path

from .base import ExportHandle, TrainResult


def export_manifest(train_result: TrainResult, export_handle: ExportHandle) -> dict:
    """Generate an export manifest from training result and export handle."""
    return {
        "source_model": train_result.model_handle.model_id,
        "adapter_dir": train_result.adapter_dir,
        "train_dir": train_result.train_dir,
        "export_dir": export_handle.export_dir,
        "export_format": export_handle.format,
        "exported_model": export_handle.model_id,
    }


def write_export_manifest(manifest: dict, output_path: Path) -> None:
    """Write export manifest to a JSON file."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
