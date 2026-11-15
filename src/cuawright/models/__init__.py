from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AnthropicMessagesAdapter": "cuawright.models.anthropic_messages",
    "ClaudeMessagesThread": "cuawright.models.anthropic_messages",
    "ModelAdapter": "cuawright.models.base",
    "ModelTurn": "cuawright.models.base",
    "OpenAIResponsesAdapter": "cuawright.models.openai_responses",
    "RUN_COMMAND_TOOL_SCHEMA": "cuawright.models.base",
    "ResponsesThread": "cuawright.models.openai_responses",
    "ToolCall": "cuawright.models.base",
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
