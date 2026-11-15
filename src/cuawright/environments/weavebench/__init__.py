from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "CUAWrightAgent",
    "CUAWrightWallTimeout",
    "WeaveBenchDesktopPlugin",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(
        import_module("cuawright.environments.weavebench.adapter"),
        name,
    )
    globals()[name] = value
    return value
