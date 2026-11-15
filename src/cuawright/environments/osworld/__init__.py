from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "GuestWorkspace": "cuawright.environments.osworld.environment",
    "OSWorldBenchmarkPlugin": "cuawright.environments.osworld.environment",
    "OSWorldEnvironmentIdentity": "cuawright.environments.osworld.lifecycle",
    "OSWorldLifecycle": "cuawright.environments.osworld.lifecycle",
    "OSWorldLifecycleConfig": "cuawright.environments.osworld.lifecycle",
    "OSWorldTaskIdentity": "cuawright.environments.osworld.lifecycle",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
