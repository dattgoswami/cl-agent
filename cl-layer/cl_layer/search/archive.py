"""Novelty archive keyed by failure signature, verifier delta, and changed files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class ArchiveKey:
    """Composite key for the novelty archive."""

    failure_signature: str
    verifier_delta: str
    changed_files: str  # frozenset serialized as sorted string

    def to_hash(self) -> str:
        raw = f"{self.failure_signature}|{self.verifier_delta}|{self.changed_files}"
        return "ark-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


class NoveltyArchive:
    """Archive that prevents near-duplicate repairs."""

    def __init__(self, window: int = 20) -> None:
        self._keys: dict[str, ArchiveKey] = {}
        self._window = window

    def is_novel(self, key: ArchiveKey) -> bool:
        """Check if a candidate is novel."""
        h = key.to_hash()
        if h in self._keys:
            return False
        # Enforce window size
        if len(self._keys) >= self._window:
            oldest = next(iter(self._keys))
            del self._keys[oldest]
        self._keys[h] = key
        return True

    def add(self, key: ArchiveKey) -> bool:
        """Add a key if novel. Returns False if duplicate."""
        if not self.is_novel(key):
            return False
        return True

    @property
    def size(self) -> int:
        return len(self._keys)

    def dedup_failure_signatures(self, signatures: list[str]) -> list[str]:
        """Deduplicate a list of failure signatures."""
        seen: set[str] = set()
        result: list[str] = []
        for sig in signatures:
            h = hashlib.sha256(sig.encode()).hexdigest()[:16]
            if h not in seen:
                seen.add(h)
                result.append(sig)
        return result
