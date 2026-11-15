from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Literal, Mapping, Protocol, Sequence, TypeAlias


RUN_COMMAND_TOOL_SCHEMA: Final[dict[str, Any]] = {
    "name": "run_command",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
        },
        "required": ["command", "timeout"],
        "additionalProperties": False,
    },
}
RUN_COMMAND_DESCRIPTION: Final = (
    "Execute a shell command in the active terminal and return its standard "
    "output, standard error, and exit status."
)
MODEL_COMMAND_OUTPUT_LIMIT = 16_000


class ModelAdapterProtocolError(RuntimeError):
    """Raised when a provider transport violates the normalized turn contract."""


class RunCommandValidationError(ValueError):
    """Raised when a tool call does not match the canonical run_command contract."""


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    data: bytes
    media_type: str
    source_path: str | None = None
    kind: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("image attachment data must be bytes")
        if not self.media_type.startswith("image/"):
            raise ValueError("image attachment media_type must start with 'image/'")


@dataclass(frozen=True, slots=True)
class FileAttachment:
    data: bytes
    media_type: str = "application/octet-stream"
    source_path: str | None = None
    kind: Literal["file"] = field(default="file", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("file attachment data must be bytes")
        if not self.media_type:
            raise ValueError("file attachment media_type cannot be empty")


Attachment: TypeAlias = ImageAttachment | FileAttachment


def cap_command_text(text: str) -> tuple[str, bool]:
    if len(text) <= MODEL_COMMAND_OUTPUT_LIMIT:
        return text, False
    notice = (
        "\n\n[OUTPUT TRUNCATED: request a narrower command, smaller range, "
        "or filtered output.]\n\n"
    )
    return text[:12_000] + notice + text[-3_000:], True


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    attachments: tuple[Attachment, ...] = ()
    completed: bool = False
    inspected: bool = False
    model_fields: Mapping[str, object] = field(default_factory=dict)
    model_output: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("exit_code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("stdout and stderr must be strings")
        attachments = tuple(self.attachments)
        if any(
            not isinstance(attachment, (ImageAttachment, FileAttachment))
            for attachment in attachments
        ):
            raise TypeError("attachments must contain typed attachment values")
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a boolean")
        if not isinstance(self.inspected, bool):
            raise TypeError("inspected must be a boolean")
        if not isinstance(self.model_fields, Mapping):
            raise TypeError("model_fields must be a mapping")
        if self.model_output is not None and not isinstance(self.model_output, str):
            raise TypeError("model_output must be a string or None")
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "model_fields", dict(self.model_fields))


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)
    parse_error: str | None = None
    raw_arguments: str | None = None

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("call_id cannot be empty")
        if not self.name:
            raise ValueError("tool name cannot be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            raise TypeError("parse_error must be a string or None")
        if self.raw_arguments is not None and not isinstance(self.raw_arguments, str):
            raise TypeError("raw_arguments must be a string or None")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True)
class ModelTurn:
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    billed_responses: int = 1
    continue_immediately: bool = False

    def __post_init__(self) -> None:
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("turn text must be a string or None")
        calls = tuple(self.tool_calls)
        if any(not isinstance(call, ToolCall) for call in calls):
            raise TypeError("tool_calls must contain ToolCall values")
        if (
            isinstance(self.billed_responses, bool)
            or not isinstance(self.billed_responses, int)
            or self.billed_responses < 1
        ):
            raise ValueError("billed_responses must be a positive integer")
        if not isinstance(self.continue_immediately, bool):
            raise TypeError("continue_immediately must be a boolean")
        object.__setattr__(self, "tool_calls", calls)


class TurnProvider(Protocol):
    def next_turn(
        self,
        *,
        tools: Sequence[Mapping[str, object]],
    ) -> ModelTurn:
        """Return the next provider response in normalized form."""

    def add_command_result(
        self,
        tool_call: ToolCall,
        result: CommandResult,
    ) -> None:
        """Add a normalized command result to the provider conversation."""

    def add_user_text(self, text: str) -> None:
        """Append one user-visible recovery or continuation message."""


class CommandExecutor(Protocol):
    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        """Execute one command and return its normalized result."""


class Compactor(Protocol):
    def __call__(
        self,
        provider: TurnProvider,
        charge_steps: Callable[[int], int],
    ) -> TurnProvider:
        """Compact one provider conversation and charge every model response."""


def normalize_run_command(call: ToolCall) -> tuple[str, int]:
    if call.name != "run_command":
        raise RunCommandValidationError(
            f"Unsupported tool call {call.name!r}; expected 'run_command'"
        )
    arguments = call.arguments
    unexpected = set(arguments) - {"command", "timeout"}
    if unexpected:
        raise RunCommandValidationError(
            "Unexpected run_command arguments: " + ", ".join(sorted(unexpected))
        )
    command = arguments.get("command")
    if not isinstance(command, str):
        raise RunCommandValidationError("run_command command must be a string")
    timeout = arguments.get("timeout")
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise RunCommandValidationError("run_command timeout must be an integer")
    if timeout < 1 or timeout > 300:
        raise RunCommandValidationError("run_command timeout must be between 1 and 300")
    return command, timeout


def canonical_tools(
    tools: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    copied = [copy.deepcopy(dict(tool)) for tool in tools]
    if copied not in ([], [RUN_COMMAND_TOOL_SCHEMA]):
        raise ModelAdapterProtocolError(
            "ModelAdapter accepts only no tools or the canonical run_command tool"
        )
    return copied


def normalized_turn(
    response: Mapping[str, Any],
    *,
    first_tool_call_only: bool,
) -> ModelTurn:
    output = response.get("output", ())
    if not isinstance(output, ABCSequence) or isinstance(output, (str, bytes)):
        raise ModelAdapterProtocolError("Provider response output must be a sequence")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in output:
        if not isinstance(item, ABCMapping):
            continue
        item_type = item.get("type")
        if item_type == "message":
            content = item.get("content", ())
            if not isinstance(content, ABCSequence) or isinstance(
                content,
                (str, bytes),
            ):
                continue
            text_parts.extend(
                str(part["text"])
                for part in content
                if isinstance(part, ABCMapping)
                and part.get("type") == "output_text"
                and part.get("text")
            )
        elif item_type == "function_call" and (
            not first_tool_call_only or not tool_calls
        ):
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise ModelAdapterProtocolError(
                    "Provider function call requires string call_id and name"
                )
            raw_arguments = item.get("arguments", "{}")
            try:
                arguments = _tool_arguments(raw_arguments)
            except ModelAdapterProtocolError as exc:
                tool_calls.append(
                    ToolCall(
                        call_id,
                        name,
                        {},
                        parse_error=str(exc),
                        raw_arguments=(
                            raw_arguments if isinstance(raw_arguments, str) else None
                        ),
                    )
                )
            else:
                tool_calls.append(ToolCall(call_id, name, arguments))

    text = "\n".join(text_parts)
    billed_responses = response.get("billed_response_count", 1)
    return ModelTurn(
        text=text or None,
        tool_calls=tuple(tool_calls),
        billed_responses=billed_responses,
        continue_immediately=response.get("continue_immediately", False),
    )


def command_result_metadata(result: CommandResult) -> dict[str, Any]:
    metadata = copy.deepcopy(dict(result.model_fields))
    status = metadata.get("status")
    if not isinstance(status, str):
        status = "success" if result.exit_code == 0 else "error"
    infrastructure_error = metadata.get("infrastructure_error", False)
    if not isinstance(infrastructure_error, bool):
        infrastructure_error = bool(infrastructure_error)
    harness = metadata.get("_harness")
    if not isinstance(harness, ABCMapping):
        harness = {}
    metadata.update(
        {
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.exit_code,
            "status": status,
            "infrastructure_error": infrastructure_error,
            "_harness": {
                **dict(harness),
                "inspected": result.inspected,
            },
            "attachments": [
                {
                    "kind": attachment.kind,
                    "media_type": attachment.media_type,
                    "source_path": attachment.source_path,
                    "size_bytes": len(attachment.data),
                    "sha256": hashlib.sha256(attachment.data).hexdigest(),
                }
                for attachment in result.attachments
            ],
        }
    )
    return metadata


def command_result_text(
    metadata: Mapping[str, Any],
    *,
    model_output: str | None = None,
) -> str:
    if model_output is not None:
        return model_output
    serialized = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return cap_command_text(serialized)[0]


def _tool_arguments(value: object) -> dict[str, object]:
    if isinstance(value, ABCMapping):
        arguments = dict(value)
    elif isinstance(value, str):
        try:
            arguments = json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise ModelAdapterProtocolError(
                f"Provider returned malformed tool arguments: {exc}"
            ) from exc
    else:
        raise ModelAdapterProtocolError(
            "Provider function call arguments must be JSON or a mapping"
        )
    if not isinstance(arguments, dict):
        raise ModelAdapterProtocolError(
            "Provider function call arguments must decode to an object"
        )
    return arguments
