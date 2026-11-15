from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "DEFAULT_BENCHMARK_COMMIT",
    "DEFAULT_DATASET_REVISION",
    "CadGenBenchManifest",
    "CadGenBenchTask",
    "discover_tasks",
    "load_manifest",
    "resolve_benchmark_commit",
    "validate_fixture_id",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(
        import_module("cuawright.environments.cadgenbench.task"),
        name,
    )
    globals()[name] = value
    return value
