"""Dataset construction package: episodes -> JSONL splits + manifest."""

from cl_layer.dataset.builder import DatasetManifest, build_dataset
from cl_layer.dataset.example_schema import ExampleType, TrainingExample, make_example_id
from cl_layer.dataset.from_episode import episode_to_example
from cl_layer.dataset.render_chat import (
    DEFAULT_CHAT_TEMPLATE,
    ChatTemplate,
    render_examples_chatl,
)
from cl_layer.dataset.splits import SplitConfig, split_datasets, split_with_config

__all__ = [
    "DatasetManifest",
    "build_dataset",
    "ChatTemplate",
    "DEFAULT_CHAT_TEMPLATE",
    "ExampleType",
    "TrainingExample",
    "make_example_id",
    "episode_to_example",
    "render_examples_chatl",
    "SplitConfig",
    "split_datasets",
    "split_with_config",
]
