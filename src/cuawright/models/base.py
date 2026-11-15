from __future__ import annotations

from typing import Final

from cuawright.runtime.contracts import (
    RUN_COMMAND_DESCRIPTION,
    RUN_COMMAND_TOOL_SCHEMA,
    CommandResult,
    ModelAdapterProtocolError,
    ModelTurn,
    ToolCall,
    TurnProvider,
    canonical_tools,
    command_result_metadata,
    command_result_text,
    normalized_turn,
)


DEFAULT_MAX_OUTPUT_TOKENS: Final = 32_768

# Compatibility alias for model adapters and adjacent benchmark integrations.
ModelAdapter = TurnProvider

__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "RUN_COMMAND_DESCRIPTION",
    "RUN_COMMAND_TOOL_SCHEMA",
    "CommandResult",
    "ModelAdapter",
    "ModelAdapterProtocolError",
    "ModelTurn",
    "ToolCall",
    "canonical_tools",
    "command_result_metadata",
    "command_result_text",
    "normalized_turn",
]
