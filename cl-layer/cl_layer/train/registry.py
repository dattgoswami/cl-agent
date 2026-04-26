"""Backend registry for trainer backends."""

from __future__ import annotations

from typing import Type

from .base import TrainerBackend

_registry: dict[str, Type[TrainerBackend]] = {}


def register_backend(name: str, backend_cls: Type[TrainerBackend]) -> None:
    """Register a trainer backend by name."""
    _registry[name] = backend_cls


def get_backend(name: str) -> Type[TrainerBackend]:
    """Get a registered backend class by name."""
    if name not in _registry:
        raise KeyError(f"Unknown backend: {name}. Registered: {list(_registry.keys())}")
    return _registry[name]


def list_backends() -> list[str]:
    """List registered backend names."""
    return list(_registry.keys())
