"""
Load OpenHands V1 conversation exports into plain Python dicts.

Supported sources
-----------------
1. ZIP export — a zip archive containing ``meta.json`` and one JSON file
   per event.
2. Filesystem directory — the persistence layout written by
   ``FilesystemEventService``:
   ``{persistence_dir}/{user_id}/v1_conversations/{conversation_id}/``
   Files are named ``{event_id_hex}.json`` (UUID hex, NOT timestamp-prefixed).
   Both loaders sort events by the ``timestamp`` field inside each event JSON,
   matching how ``EventServiceBase.search_events`` orders results.

No OpenHands runtime dependency is required.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpenHandsConversation:
    """
    A V1 OpenHands conversation loaded into plain Python dicts.

    ``meta``   — parsed ``meta.json`` (may be ``{}`` if absent).
    ``events`` — list of raw event dicts sorted by ``event["timestamp"]``
                 in ascending order.
    ``source_path`` — string path to the zip or directory, for tracing.
    """

    meta: dict
    events: list[dict]
    source_path: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_json_bytes(data: bytes, origin: str) -> dict:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse JSON from {origin!r}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object in {origin!r}, got {type(parsed).__name__}")
    return parsed


def _is_event_file(name: str) -> bool:
    """Return True for per-event JSON files (not meta.json)."""
    lower = name.lower()
    return lower.endswith(".json") and "meta" not in lower.split("/")[-1].split(".")[0].lower()


# ---------------------------------------------------------------------------
# ZIP loader
# ---------------------------------------------------------------------------

def load_conversation_zip(path: str | Path) -> OpenHandsConversation:
    """Load a V1 conversation from a ZIP export file.

    The archive must contain ``meta.json`` (optional) and one or more
    event JSON files.  Events are sorted by their ``timestamp`` field after
    parsing, matching the order produced by ``EventServiceBase.search_events``.
    Malformed event JSON raises ``ValueError``; importer callers should not
    silently train on partial conversation data.
    """
    zip_path = Path(path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    meta: dict = {}
    event_items: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            basename = info.filename.split("/")[-1]
            if not basename:
                continue
            if basename.lower() == "meta.json":
                meta = _parse_json_bytes(zf.read(info.filename), info.filename)
            elif _is_event_file(basename):
                event_items.append((basename, zf.read(info.filename)))

    events: list[dict] = []
    for name, data in event_items:
        events.append(_parse_json_bytes(data, name))
    events.sort(key=lambda ev: ev.get("timestamp") or "")

    return OpenHandsConversation(
        meta=meta,
        events=events,
        source_path=str(zip_path),
    )


# ---------------------------------------------------------------------------
# Filesystem directory loader
# ---------------------------------------------------------------------------

def load_conversation_dir(path: str | Path) -> OpenHandsConversation:
    """Load a V1 conversation from a filesystem persistence directory.

    The directory should contain ``meta.json`` (optional) and one
    ``{event_id_hex}.json`` file per event (UUID hex filename, as written by
    ``FilesystemEventService``).  Events are sorted by their ``timestamp``
    field after parsing, matching the order produced by
    ``EventServiceBase.search_events``.
    Malformed event JSON raises ``ValueError``; importer callers should not
    silently train on partial conversation data.
    """
    dir_path = Path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Conversation directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    meta: dict = {}
    meta_file = dir_path / "meta.json"
    if meta_file.is_file():
        meta = _parse_json_bytes(meta_file.read_bytes(), str(meta_file))

    event_files = [f for f in dir_path.iterdir() if f.is_file() and _is_event_file(f.name)]

    events: list[dict] = []
    for f in event_files:
        events.append(_parse_json_bytes(f.read_bytes(), str(f)))
    events.sort(key=lambda ev: ev.get("timestamp") or "")

    return OpenHandsConversation(
        meta=meta,
        events=events,
        source_path=str(dir_path),
    )


# ---------------------------------------------------------------------------
# Auto-detect loader
# ---------------------------------------------------------------------------

def load_conversation(path: str | Path) -> OpenHandsConversation:
    """Load a conversation from a ZIP file or filesystem directory.

    Detects the source type from whether *path* is a ``.zip`` file or a
    directory.  Raises ``ValueError`` if neither applies.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        return load_conversation_zip(p)
    if p.is_dir():
        return load_conversation_dir(p)
    raise ValueError(
        f"Cannot load conversation from {p!r}: must be a .zip file or a directory"
    )


# ---------------------------------------------------------------------------
# Helper: create an in-memory ZIP for testing
# ---------------------------------------------------------------------------

def build_conversation_zip(
    events: list[dict],
    meta: dict | None = None,
) -> bytes:
    """Produce a valid V1 conversation ZIP from Python dicts (test helper).

    Returns raw ZIP bytes that can be written to a tmp_path file before
    calling :func:`load_conversation_zip`.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if meta is not None:
            zf.writestr("meta.json", json.dumps(meta))
        for idx, ev in enumerate(events):
            zf.writestr(f"event_{idx:06d}_{ev.get('id', f'ev{idx}')}.json", json.dumps(ev))
    return buf.getvalue()
