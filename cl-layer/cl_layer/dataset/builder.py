"""End-to-end dataset orchestrator.

Consumes a stream of episodes, runs convert → filter → dedup → split →
render-as-JSONL, writes ``train.jsonl`` / ``valid.jsonl`` / ``test.jsonl``
and a ``manifest.json`` under ``output_root/<gen_id>/``, and returns a
``DatasetManifest``.

This is the single source of truth for "how a generation's training set is
materialized." The trainer reads JSONL produced here; the manifest is the
audit trail (source episodes, split sizes, filter/dedup counts, template
config) for promotion review.

No optional/heavyweight dependencies are imported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from cl_layer.dataset.dedup import dedup_examples
from cl_layer.dataset.example_schema import TrainingExample
from cl_layer.dataset.filters import filter_examples
from cl_layer.dataset.from_episode import episode_to_example
from cl_layer.dataset.render_chat import ChatTemplate, render_examples_chatl
from cl_layer.dataset.splits import SplitConfig, split_with_config
from cl_layer.episode.schema import Episode


@dataclass
class SplitInfo:
    name: str
    size: int
    path: str
    example_ids: list[str] = field(default_factory=list)


@dataclass
class DatasetManifest:
    """Audit record for one generation's materialized dataset."""

    gen_id: str
    output_dir: str
    created_at: str
    template: dict
    source_episode_ids: list[str]
    counts: dict[str, int]
    splits: dict[str, dict]
    split_config: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _template_to_dict(template: ChatTemplate) -> dict:
    return {
        "role_start": template.role_start,
        "role_end": template.role_end,
        "system_prompt": template.system_prompt,
    }


def _write_jsonl(lines: list[str], path: Path) -> None:
    if lines:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def build_dataset(
    episodes: Iterable[Episode],
    output_root: str | Path,
    gen_id: str,
    template: ChatTemplate | None = None,
    split_config: SplitConfig | None = None,
) -> DatasetManifest:
    """Materialize a JSONL dataset and manifest for one generation.

    Layout written under ``output_root/<gen_id>/``:
        - ``train.jsonl`` — one JSON object per line, ``{"messages": [...]}``
        - ``valid.jsonl``
        - ``test.jsonl``
        - ``manifest.json``

    Returns a ``DatasetManifest`` describing what was written.
    """
    out_dir = Path(output_root) / gen_id
    out_dir.mkdir(parents=True, exist_ok=True)

    template = template or ChatTemplate()
    split_cfg = split_config or SplitConfig()

    # 1. Convert episodes → eligible examples.
    converted: list[TrainingExample] = []
    accepted_episode_ids: list[str] = []
    rejected_episodes = 0
    for ep in episodes:
        ex = episode_to_example(ep)
        if ex is None:
            rejected_episodes += 1
            continue
        converted.append(ex)
        accepted_episode_ids.append(ep.episode_id)

    # 2. Filter.
    filtered, filter_counts = filter_examples(converted)

    # 3. Dedup (patch hash + normalized text).
    deduped, dedup_counts = dedup_examples(filtered)

    # 4. Split, grouping by task_id.
    train, valid, test = split_with_config(deduped, split_cfg)

    # 5. Render and write JSONL for each split.
    splits_meta: dict[str, dict] = {}
    for name, split_examples in (("train", train), ("valid", valid), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        lines = render_examples_chatl(split_examples, template)
        _write_jsonl(lines, path)
        splits_meta[name] = SplitInfo(
            name=name,
            size=len(split_examples),
            path=str(path),
            example_ids=[ex.id for ex in split_examples],
        ).__dict__

    # 6. Manifest.
    counts: dict[str, int] = {
        "episodes_seen": len(accepted_episode_ids) + rejected_episodes,
        "episodes_rejected": rejected_episodes,
        "examples_converted": len(converted),
        "examples_after_filter": filter_counts.get("output", len(filtered)),
        "examples_after_dedup": dedup_counts.get("after_normalized_text", len(deduped)),
        "examples_total_in_splits": len(train) + len(valid) + len(test),
        "rejected_no_verifier": filter_counts.get("rejected_no_verifier", 0),
        "rejected_empty_target": filter_counts.get("rejected_empty_target", 0),
        "rejected_giant_diff": filter_counts.get("rejected_giant_diff", 0),
        "rejected_hidden_state": filter_counts.get("rejected_hidden_state", 0),
        "dedup_after_patch_hash": dedup_counts.get("after_patch_hash", len(filtered)),
    }

    manifest = DatasetManifest(
        gen_id=gen_id,
        output_dir=str(out_dir),
        created_at=datetime.now(timezone.utc).isoformat(),
        template=_template_to_dict(template),
        source_episode_ids=accepted_episode_ids,
        counts=counts,
        splits=splits_meta,
        split_config={
            "train_ratio": split_cfg.train_ratio,
            "valid_ratio": split_cfg.valid_ratio,
            "test_ratio": split_cfg.test_ratio,
            "seed": split_cfg.seed,
        },
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest
