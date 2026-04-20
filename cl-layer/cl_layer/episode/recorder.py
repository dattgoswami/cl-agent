from __future__ import annotations

import json
from pathlib import Path

from .schema import Episode, episode_from_dict, episode_to_dict


class EpisodeRecorder:
    """Append-only JSONL episode store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, episode: Episode) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(episode_to_dict(episode))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_all(self) -> list[Episode]:
        if not self.path.exists():
            return []
        episodes: list[Episode] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(episode_from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    # skip malformed lines rather than crashing the session
                    pass
        return episodes
