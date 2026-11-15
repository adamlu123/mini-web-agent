from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AgentLoop": "cuawright.agents.default",
    "AgentLoopResult": "cuawright.agents.default",
    "TerminalAgentRuntime": "cuawright.agents.orchestration",
    "TerminalRuntimeResult": "cuawright.agents.orchestration",
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
