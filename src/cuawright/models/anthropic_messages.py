from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import io
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from anthropic import APIStatusError, transform_schema
from PIL import Image, ImageOps

from cuawright.artifacts import TraceWriter
from cuawright.models.anthropic_client import (
    ClaudeMessagesClient,
    ClaudeStopError,
    ClaudeStreamParseError,
    build_create_request,
    serialized_request_bytes,
)
from cuawright.models.anthropic_cost import AnthropicCostTracker
from cuawright.models.base import (
    RUN_COMMAND_DESCRIPTION,
    ModelAdapterProtocolError,
    canonical_tools,
    command_result_metadata,
    command_result_text,
    normalized_turn,
)
from cuawright.models.openai_responses import openai_responses_tools
from cuawright.models.openrouter_cost import OpenRouterCostTracker
from cuawright.models.base import ModelTurn, ToolCall
from cuawright.runtime.contracts import CommandResult, ImageAttachment


ClaudeProvider = Literal["anthropic", "openrouter-anthropic"]
OPENROUTER_CLAUDE_MAX_IMAGE_DIMENSION = 1_500
OPENROUTER_CLAUDE_MAX_IMAGE_BYTES = 400_000
ANTHROPIC_MAX_IMAGE_DIMENSION = 1_920
ANTHROPIC_MAX_IMAGE_BYTES = 1_500_000
ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES = 3
ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL = 4
ANTHROPIC_DEFAULT_MAX_REQUEST_MB = 24.0
ANTHROPIC_MAX_CONFIGURABLE_REQUEST_MB = 32.0
CLAUDE_DESKTOP_SIZE = (1_920, 1_080)
CLAUDE_IMAGE_PLACEHOLDER = "[Image Omitted]"
ANTHROPIC_TRANSIENT_RETRY_DELAYS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
ANTHROPIC_POLICY_ERROR_MARKERS = (
    "content filter",
    "content filtering policy",
    "content_policy",
    "cyber_policy",
    "usage policy",
    "policy violation",
)
CLAUDE_PARALLEL_TOOL_PROMPT = """
You may call tools in parallel when they are independent and non-conflicting.
Do not parallelize calls that can race through shared state or depend on one
another's results; when uncertain, call one tool and wait.
""".strip()


class ClaudeRequestTooLargeError(RuntimeError):
    pass


def anthropic_request_limit_bytes(
    min_recent_images: int,
    image_prune_interval: int,
    max_request_mb: float,
) -> int:
    if type(min_recent_images) is not int:
        raise TypeError("anthropic_min_recent_images must be an integer")
    if min_recent_images < 0:
        raise ValueError("anthropic_min_recent_images cannot be negative")
    if type(image_prune_interval) is not int:
        raise TypeError("anthropic_image_prune_interval must be an integer")
    if image_prune_interval < 1:
        raise ValueError("anthropic_image_prune_interval must be positive")
    if isinstance(max_request_mb, bool) or not isinstance(
        max_request_mb,
        (int, float),
    ):
        raise TypeError("anthropic_max_request_mb must be a number")
    if not math.isfinite(max_request_mb) or max_request_mb <= 0:
        raise ValueError("anthropic_max_request_mb must be a positive finite number")
    if max_request_mb > ANTHROPIC_MAX_CONFIGURABLE_REQUEST_MB:
        raise ValueError(
            "anthropic_max_request_mb cannot exceed Anthropic's 32 MB "
            "request-body limit"
        )
    limit_bytes = round(max_request_mb * 1_000_000)
    if limit_bytes < 1:
        raise ValueError("anthropic_max_request_mb must allow at least one byte")
    return limit_bytes


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _anthropic_policy_api_error_category(exc: APIStatusError) -> str | None:
    body = getattr(exc, "body", None)
    diagnostic = f"{exc} {json.dumps(body, default=str)}".lower()
    if not any(marker in diagnostic for marker in ANTHROPIC_POLICY_ERROR_MARKERS):
        return None
    if "cyber" in diagnostic:
        return "cyber_policy"
    if "content" in diagnostic:
        return "content_filter"
    return "policy"


def _anthropic_policy_category(value: Any) -> str | None:
    category = (
        value.get("category")
        if isinstance(value, Mapping)
        else getattr(value, "category", None)
    )
    if not isinstance(category, str):
        return None
    category = category.strip().lower()
    return category or None


def _anthropic_policy_error_summary(value: Any) -> str | None:
    if isinstance(value, Mapping):
        nested = value.get("error")
        if nested is not None:
            summary = _anthropic_policy_error_summary(nested)
            if summary:
                return summary
        candidates = (
            value.get("message"),
            value.get("explanation"),
            value.get("detail"),
        )
    else:
        candidates = (
            getattr(value, "message", None),
            getattr(value, "explanation", None),
            getattr(value, "detail", None),
        )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return " ".join(candidate.split())[:1_000]
    if isinstance(value, Exception):
        summary = " ".join(str(value).split())
        return summary[:1_000] if summary else None
    return None


def _anthropic_policy_feedback(
    *,
    attempt: int,
    maximum: int,
    category: str | None,
    error_summary: str | None,
) -> str:
    provider_error = error_summary or category or "unspecified policy block"
    return (
        "The provider rejected your immediately preceding response with this "
        f"policy error (compliance retry {attempt}/{maximum}): {provider_error} "
        "Do not repeat, disguise, or work around the blocked action. Reassess the "
        "task and continue only with a clearly benign, policy-compliant approach "
        "using the existing tools. Prefer ordinary user-visible application "
        "operations and avoid security-sensitive commands or techniques. If no "
        "compliant continuation is possible, explain that briefly instead of "
        "attempting the blocked action."
    )


class PreparedImage(Protocol):
    media_type: str
    data: bytes
    dimension_metadata: dict[str, Any] | None


class ClaudeMessagesThreadProtocol(Protocol):
    provider: ClaudeProvider

    def call(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def add_tool_outputs(self, outputs: list[tuple[str, Any]]) -> None: ...

    def prepare_image(self, mime: str, data: bytes) -> PreparedImage: ...

    def add_user_text(self, text: str) -> None: ...


def translate_anthropic_tools(
    tools: Sequence[Mapping[str, Any]],
    *,
    direct: bool,
) -> list[dict[str, Any]]:
    """Translate Responses-style tools to exact Messages API tool objects."""

    converted: list[dict[str, Any]] = []
    for tool in tools:
        parameters = tool.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ModelAdapterProtocolError(
                "Anthropic tool translation requires an object parameters schema"
            )
        converted_tool = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": (
                transform_schema(copy.deepcopy(dict(parameters)))
                if direct
                else copy.deepcopy(dict(parameters))
            ),
        }
        if tool.get("strict") is True:
            converted_tool["strict"] = True
        converted.append(converted_tool)
    return converted


def anthropic_messages_tools(
    tools: Sequence[Mapping[str, object]],
    *,
    direct: bool,
) -> list[dict[str, Any]]:
    """Translate the frozen core schema to native Messages tool form."""

    canonical = canonical_tools(tools)
    responses_tools = [
        {
            "name": tool["name"],
            "description": RUN_COMMAND_DESCRIPTION,
            "parameters": copy.deepcopy(tool["parameters"]),
            "strict": tool.get("strict") is True,
        }
        for tool in canonical
    ]
    return translate_anthropic_tools(responses_tools, direct=direct)


class AnthropicMessagesAdapter:
    """Compatibility adapter retaining native Claude batching and image policy."""

    def __init__(
        self,
        thread: ClaudeMessagesThreadProtocol,
        *,
        provider: ClaudeProvider | None = None,
    ) -> None:
        self.thread = thread
        self.provider = provider or thread.provider
        if self.provider not in {"anthropic", "openrouter-anthropic"}:
            raise ValueError(f"Unsupported Claude provider: {self.provider!r}")
        self._pending_calls: tuple[ToolCall, ...] = ()
        self._pending_outputs: list[tuple[str, Any]] = []

    def next_turn(
        self,
        *,
        tools: Sequence[Mapping[str, object]],
    ) -> ModelTurn:
        if self._pending_calls:
            raise ModelAdapterProtocolError(
                "Cannot request another Claude turn before all tool results are added"
            )

        provider_tools = anthropic_messages_tools(
            tools,
            direct=self.provider == "anthropic",
        )
        call_provider_tools = getattr(self.thread, "call_provider_tools", None)
        if callable(call_provider_tools):
            response = call_provider_tools(tools=provider_tools)
        else:
            response = self.thread.call(tools=openai_responses_tools(tools))
        if not isinstance(response, Mapping):
            raise TypeError("ClaudeMessagesThread.call() must return a mapping")

        turn = normalized_turn(response, first_tool_call_only=False)
        call_ids = [call.call_id for call in turn.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ModelAdapterProtocolError(
                "Claude returned duplicate tool call identifiers"
            )
        self._pending_calls = turn.tool_calls
        return turn

    def add_command_result(
        self,
        tool_call: ToolCall,
        result: CommandResult,
    ) -> None:
        if not self._pending_calls:
            raise ModelAdapterProtocolError(
                "Claude command result does not match a pending tool call"
            )
        completed = len(self._pending_outputs)
        expected = self._pending_calls[completed]
        if tool_call.call_id != expected.call_id:
            raise ModelAdapterProtocolError(
                "Claude command results must be added in tool-call order"
            )

        self._pending_outputs.append((tool_call.call_id, self._command_output(result)))
        if len(self._pending_outputs) == len(self._pending_calls):
            self.thread.add_tool_outputs(self._pending_outputs)
            self._pending_calls = ()
            self._pending_outputs = []

    def _command_output(self, result: CommandResult) -> Any:
        metadata = command_result_metadata(result)
        content: list[dict[str, Any]] = []
        for index, attachment in enumerate(result.attachments):
            if not isinstance(attachment, ImageAttachment):
                continue
            prepared = self.thread.prepare_image(
                attachment.media_type,
                attachment.data,
            )
            geometry = prepared.dimension_metadata
            if geometry is not None:
                metadata["attachments"][index]["image_geometry"] = geometry
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": prepared.media_type,
                        "data": base64.b64encode(prepared.data).decode("ascii"),
                    },
                }
            )
        if not content:
            return command_result_text(metadata, model_output=result.model_output)
        return [
            {
                "type": "text",
                "text": command_result_text(
                    metadata,
                    model_output=result.model_output,
                ),
            },
            *content,
        ]

    def add_user_text(self, text: str) -> None:
        if self._pending_calls:
            raise ModelAdapterProtocolError(
                "Cannot append user text before all Claude tool results are added"
            )
        self.thread.add_user_text(text)


def _anthropic_tools(
    tools: list[dict[str, Any]],
    *,
    direct: bool = False,
) -> list[dict[str, Any]]:
    return translate_anthropic_tools(tools, direct=direct)


@dataclass(frozen=True)
class ClaudePreparedImage:
    media_type: str
    data: bytes
    original_size: tuple[int, int]
    sent_size: tuple[int, int]

    @property
    def dimension_metadata(self) -> dict[str, Any] | None:
        if min(*self.original_size, *self.sent_size) < 1:
            return None
        if (
            self.original_size == self.sent_size
            and self.sent_size == CLAUDE_DESKTOP_SIZE
        ):
            return None
        original_width, original_height = self.original_size
        sent_width, sent_height = self.sent_size
        desktop_width, desktop_height = CLAUDE_DESKTOP_SIZE
        displayed_to_source_x = original_width / sent_width
        displayed_to_source_y = original_height / sent_height
        displayed_to_desktop_x = desktop_width / sent_width
        displayed_to_desktop_y = desktop_height / sent_height
        return {
            "source_dimensions": {
                "width": original_width,
                "height": original_height,
            },
            "displayed_dimensions": {
                "width": sent_width,
                "height": sent_height,
            },
            "desktop_dimensions": {
                "width": desktop_width,
                "height": desktop_height,
            },
            "displayed_to_source_scale": {
                "x": round(displayed_to_source_x, 6),
                "y": round(displayed_to_source_y, 6),
            },
            "displayed_to_desktop_scale": {
                "x": round(displayed_to_desktop_x, 6),
                "y": round(displayed_to_desktop_y, 6),
            },
            "coordinate_guidance": (
                "The VM desktop coordinate space is 1920x1080. If this image is "
                "a full-desktop screenshot, multiply displayed-image x/y "
                f"coordinates by {displayed_to_desktop_x:.6f}/"
                f"{displayed_to_desktop_y:.6f} before using xdotool or "
                "PyAutoGUI. For a crop or non-desktop image, do not treat its "
                "pixels as desktop coordinates."
            ),
        }


def _prepare_claude_image(
    mime: str,
    data: bytes,
    *,
    direct: bool,
) -> ClaudePreparedImage:
    max_dimension = (
        ANTHROPIC_MAX_IMAGE_DIMENSION
        if direct
        else OPENROUTER_CLAUDE_MAX_IMAGE_DIMENSION
    )
    max_bytes = (
        ANTHROPIC_MAX_IMAGE_BYTES if direct else OPENROUTER_CLAUDE_MAX_IMAGE_BYTES
    )
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)
        original_size = image.size
        if max(original_size) <= max_dimension and len(data) <= max_bytes:
            return ClaudePreparedImage(mime, data, original_size, original_size)
        image.thumbnail(
            (max_dimension, max_dimension),
            Image.Resampling.LANCZOS,
        )
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        qualities = (92, 88, 82, 76, 68, 58) if direct else (88, 78, 68, 58, 48)
        for quality in qualities:
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            if len(encoded) <= max_bytes:
                return ClaudePreparedImage(
                    "image/jpeg",
                    encoded,
                    original_size,
                    image.size,
                )
        shrink_factor = 0.9 if direct else 0.8
        while len(encoded) > max_bytes and image.size != (1, 1):
            next_size = (
                max(1, round(image.width * shrink_factor)),
                max(1, round(image.height * shrink_factor)),
            )
            if next_size == image.size:
                next_size = (
                    max(1, image.width - 1),
                    max(1, image.height - 1),
                )
            image.thumbnail(next_size, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=58, optimize=True)
            encoded = output.getvalue()
        if len(encoded) > max_bytes:
            raise ValueError(
                f"Claude image could not be compressed below {max_bytes} bytes"
            )
        return ClaudePreparedImage(
            "image/jpeg",
            encoded,
            original_size,
            image.size,
        )


def _claude_image_payload(
    mime: str,
    data: bytes,
    *,
    direct: bool = False,
) -> tuple[str, bytes]:
    prepared = _prepare_claude_image(mime, data, direct=direct)
    return prepared.media_type, prepared.data


def _claude_image_blocks(
    prepared: ClaudePreparedImage,
    description: dict[str, Any],
) -> list[dict[str, Any]]:
    payload = copy.deepcopy(description)
    payload.pop("image_geometry", None)
    if prepared.dimension_metadata is not None:
        payload["image_geometry"] = prepared.dimension_metadata
    return [
        {
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=False),
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": prepared.media_type,
                "data": base64.b64encode(prepared.data).decode("ascii"),
            },
        },
    ]


def _is_claude_base64_image_block(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("type") != "image":
        return False
    source = value.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        return False
    if source.get("media_type") not in {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        return False
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return False
    try:
        base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        return False
    return True


def _claude_image_slots(messages: Any) -> list[tuple[list[Any], int]]:
    """Find valid image blocks only in Anthropic message/tool-result content."""
    slots: list[tuple[list[Any], int]] = []

    def visit_content(content: Any) -> None:
        if not isinstance(content, list):
            return
        for index, block in enumerate(content):
            if _is_claude_base64_image_block(block):
                slots.append((content, index))
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                visit_content(block.get("content"))

    if not isinstance(messages, list):
        return slots
    for message in messages:
        if isinstance(message, dict) and message.get("role") in {
            "assistant",
            "user",
        }:
            visit_content(message.get("content"))
    return slots


def _prune_claude_images(value: Any, count: int) -> int:
    slots = _claude_image_slots(value)
    prune_count = min(max(count, 0), len(slots))
    for container, key in slots[:prune_count]:
        container[key] = {
            "type": "text",
            "text": CLAUDE_IMAGE_PLACEHOLDER,
        }
    return prune_count


def _claude_replay_content(
    content: list[Any],
    *,
    direct: bool = False,
) -> list[Any]:
    if direct:
        # Official SDK response blocks are valid request params and retain opaque
        # compaction metadata plus thinking signatures when round-tripped.
        return list(content)
    replay: list[Any] = []
    for block in content:
        if getattr(block, "type", None) != "compaction":
            replay.append(block)
            continue
        payload = block.model_dump(mode="json", exclude_none=True)
        if payload.get("encrypted_content") is None:
            payload.pop("encrypted_content", None)
        replay.append(payload)
    return replay


class ClaudeMessagesThread:
    """Synchronous thread adapter for Claude's native Messages API."""

    def __init__(
        self,
        *,
        client: ClaudeMessagesClient,
        trace: TraceWriter,
        role: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        cost_tracker: OpenRouterCostTracker | AnthropicCostTracker,
        provider: ClaudeProvider,
        async_runner: asyncio.Runner | None = None,
        provider_preferences: Mapping[str, Any] | None = None,
        compaction_reasoning_effort: str | None = None,
        anthropic_min_recent_images: int = ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES,
        anthropic_image_prune_interval: int = ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL,
        anthropic_max_request_mb: float = ANTHROPIC_DEFAULT_MAX_REQUEST_MB,
    ) -> None:
        self.client = client
        self.trace = trace
        self.role = role
        self.system_prompt = system_prompt
        if (
            not client.config.disable_parallel_tool_use
            and CLAUDE_PARALLEL_TOOL_PROMPT not in system_prompt
        ):
            self.system_prompt = f"{system_prompt}\n\n{CLAUDE_PARALLEL_TOOL_PROMPT}"
        self.tools = tools
        self.max_output_tokens = max_output_tokens
        self.cost_tracker = cost_tracker
        self.async_runner = async_runner
        if provider not in {"anthropic", "openrouter-anthropic"}:
            raise ValueError(f"Unsupported Claude provider: {provider!r}")
        self.provider = provider
        self.provider_preferences = dict(provider_preferences or {})
        self.compaction_reasoning_effort = compaction_reasoning_effort
        if self.provider_preferences and provider != "openrouter-anthropic":
            raise ValueError(
                "provider_preferences require provider='openrouter-anthropic'"
            )
        self.anthropic_min_recent_images = anthropic_min_recent_images
        self.anthropic_image_prune_interval = anthropic_image_prune_interval
        self.anthropic_max_request_bytes = anthropic_request_limit_bytes(
            anthropic_min_recent_images,
            anthropic_image_prune_interval,
            anthropic_max_request_mb,
        )
        self._consecutive_pause_turns = 0
        self._policy_feedback_retries_used = 0
        self.session_id = f"osworld-cli-{role}-{uuid.uuid4().hex}"
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

    def _policy_feedback_recovery(
        self,
        *,
        source: str,
        category: str | None,
        error_summary: str | None,
        billed_response_count: int,
        rollback_assistant: bool,
    ) -> dict[str, Any] | None:
        maximum = self.client.config.policy_feedback_max_retries
        if (
            self.provider != "anthropic"
            or self._policy_feedback_retries_used >= maximum
        ):
            return None
        if rollback_assistant:
            if not self.messages or self.messages[-1].get("role") != "assistant":
                raise RuntimeError(
                    "Anthropic policy feedback could not roll back assistant history"
                )
            self.messages.pop()
        self._policy_feedback_retries_used += 1
        feedback = _anthropic_policy_feedback(
            attempt=self._policy_feedback_retries_used,
            maximum=maximum,
            category=category,
            error_summary=error_summary,
        )
        self.messages.append({"role": "user", "content": feedback})
        self.trace.event(
            "anthropic_policy_feedback_retry",
            {
                "role": self.role,
                "attempt": self._policy_feedback_retries_used,
                "max_attempts": maximum,
                "source": source,
                "category": category,
                "billed_response_count": billed_response_count,
                "feedback": feedback,
            },
        )
        return {
            "output": [],
            "stop_reason": "policy_feedback_retry",
            "billed_response_count": billed_response_count,
            "continue_immediately": True,
            "synthetic_recovery": True,
            "policy_feedback_retry": True,
        }

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_output(self, call_id: str, output: Any) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output,
                    }
                ],
            }
        )

    def add_tool_outputs(self, outputs: list[tuple[str, Any]]) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output,
                    }
                    for call_id, output in outputs
                ],
            }
        )

    def add_image(self, path: str, mime: str, data: bytes) -> None:
        prepared = self.prepare_image(mime, data)
        description: dict[str, Any] = {
            "status": "success",
            "image_attached": path,
        }
        self.messages.append(
            {
                "role": "user",
                "content": _claude_image_blocks(prepared, description),
            }
        )

    def prepare_image(self, mime: str, data: bytes) -> ClaudePreparedImage:
        return _prepare_claude_image(
            mime,
            data,
            direct=self.provider == "anthropic",
        )

    def _apply_direct_request_policy(
        self,
        request: dict[str, Any],
    ) -> None:
        request_messages = request.get("messages")
        if not isinstance(request_messages, list):
            raise TypeError("Claude request messages must be a list")
        before_size = serialized_request_bytes(request)
        full_serializations = 1
        total_images = len(_claude_image_slots(request_messages))
        prunable_images = max(
            0,
            total_images - self.anthropic_min_recent_images,
        )
        periodic_pruned = (
            prunable_images // self.anthropic_image_prune_interval
        ) * self.anthropic_image_prune_interval
        if periodic_pruned:
            pruned = _prune_claude_images(request_messages, periodic_pruned)
            if pruned != periodic_pruned:
                raise RuntimeError(
                    "Claude image-pruning state became inconsistent during "
                    "request preflight"
                )
            after_size = serialized_request_bytes(request)
            full_serializations += 1
        else:
            after_size = before_size

        forced_pruned_images = 0
        if after_size > self.anthropic_max_request_bytes:
            remaining_slots = _claude_image_slots(request_messages)
            max_forced = max(
                0,
                len(remaining_slots) - self.anthropic_min_recent_images,
            )
            required_savings = after_size - self.anthropic_max_request_bytes
            placeholder = {
                "type": "text",
                "text": CLAUDE_IMAGE_PLACEHOLDER,
            }
            estimated_savings = 0
            for container, key in remaining_slots[:max_forced]:
                estimated_savings += serialized_request_bytes(
                    {"block": container[key]}
                ) - serialized_request_bytes({"block": placeholder})
                forced_pruned_images += 1
                if estimated_savings >= required_savings:
                    break
            if forced_pruned_images:
                pruned = _prune_claude_images(
                    request_messages,
                    forced_pruned_images,
                )
                if pruned != forced_pruned_images:
                    raise RuntimeError(
                        "Claude image-pruning state became inconsistent during "
                        "request preflight"
                    )
                after_size = serialized_request_bytes(request)
                full_serializations += 1
            if (
                forced_pruned_images < max_forced
                and after_size > self.anthropic_max_request_bytes
            ):
                raise RuntimeError(
                    "Claude request-size estimate became inconsistent during "
                    "image pruning"
                )

        blocked = after_size > self.anthropic_max_request_bytes
        retained_images = len(_claude_image_slots(request_messages))
        pruned_images = periodic_pruned + forced_pruned_images
        # Preserve pruning in model-visible history without aliasing its outer
        # list to the sent request, which must remain immutable for tracing.
        self.messages = list(request_messages)
        self.trace.event(
            "anthropic_request_preflight",
            {
                "role": self.role,
                "status": "blocked" if blocked else "ready",
                "before_bytes": before_size,
                "after_bytes": after_size,
                "limit_bytes": self.anthropic_max_request_bytes,
                "total_images": total_images,
                "pruned_images": pruned_images,
                "periodic_pruned_images": periodic_pruned,
                "retained_images": retained_images,
                "forced_pruned_images": forced_pruned_images,
                "forced_to_minimum": (
                    forced_pruned_images > 0
                    and retained_images == self.anthropic_min_recent_images
                ),
                "min_recent_images": self.anthropic_min_recent_images,
                "prune_interval": self.anthropic_image_prune_interval,
                "full_serializations": full_serializations,
            },
        )
        if blocked:
            raise ClaudeRequestTooLargeError(
                "Direct Anthropic request is "
                f"{after_size / 1_000_000:.2f} MB after pruning "
                f"{pruned_images} of {total_images} image(s) "
                f"({retained_images} recent image(s) retained), above "
                f"the configured {self.anthropic_max_request_bytes / 1_000_000:.2f} "
                "MB local limit; no API request was sent"
            )

    def call(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        _provider_tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        active_tools = self.tools if tools is None else tools
        provider_tools = None if _provider_tools is None else list(_provider_tools)
        claude_tool_choice: dict[str, Any] | None = None
        if tool_choice == "none":
            active_tools = []
            provider_tools = []
        elif tool_choice == "required":
            claude_tool_choice = {
                "type": "any",
                "disable_parallel_tool_use": (
                    self.client.config.disable_parallel_tool_use
                ),
            }
        elif isinstance(tool_choice, dict):
            forced_name = tool_choice.get("name")
            if forced_name:
                claude_tool_choice = {
                    "type": "tool",
                    "name": forced_name,
                    "disable_parallel_tool_use": (
                        self.client.config.disable_parallel_tool_use
                    ),
                }
        elif provider_tools or active_tools:
            claude_tool_choice = {
                "type": "auto",
                "disable_parallel_tool_use": (
                    self.client.config.disable_parallel_tool_use
                ),
            }

        converted_tools = (
            provider_tools
            if provider_tools is not None
            else _anthropic_tools(
                active_tools,
                direct=self.provider == "anthropic",
            )
        )
        extra_body: dict[str, Any] = {}
        if self.provider == "openrouter-anthropic":
            extra_body["session_id"] = self.session_id
        if self.provider_preferences:
            extra_body["provider"] = self.provider_preferences
        self.cost_tracker.ensure_request_allowed()
        request = build_create_request(
            self.client.config,
            messages=self.messages,
            system=self.system_prompt,
            tools=converted_tools or None,
            tool_choice=claude_tool_choice,
            max_tokens=max_output_tokens or self.max_output_tokens,
            extra_body=extra_body or None,
        )
        if reasoning_effort is not None:
            request["output_config"] = {"effort": reasoning_effort}
        if self.provider == "anthropic":
            self._apply_direct_request_policy(request)
        try:
            transient_attempt = 0
            while True:
                coroutine = self.client.create_prepared_turn(request=request)
                try:
                    turn = (
                        self.async_runner.run(coroutine)
                        if self.async_runner is not None
                        else asyncio.run(coroutine)
                    )
                    break
                except APIStatusError as exc:
                    policy_category = _anthropic_policy_api_error_category(exc)
                    retryable = (
                        exc.status_code in {408, 409, 429}
                        or 500 <= exc.status_code < 600
                    )
                    if policy_category is not None:
                        billed_response_count = getattr(
                            exc,
                            "billed_response_count",
                            1,
                        )
                        if (
                            isinstance(billed_response_count, bool)
                            or not isinstance(billed_response_count, int)
                            or billed_response_count < 1
                        ):
                            billed_response_count = 1
                        self.trace.model_call(
                            _jsonable(request),
                            {
                                "status": "policy_blocked",
                                "output": [],
                                "stop_reason": "policy_error",
                                "policy_category": policy_category,
                                "billed_response_count": billed_response_count,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        recovery = self._policy_feedback_recovery(
                            source="api_status_error",
                            category=policy_category,
                            error_summary=_anthropic_policy_error_summary(exc),
                            billed_response_count=billed_response_count,
                            rollback_assistant=False,
                        )
                        if recovery is not None:
                            return recovery
                    if not retryable or transient_attempt >= len(
                        ANTHROPIC_TRANSIENT_RETRY_DELAYS
                    ):
                        raise
                    delay = ANTHROPIC_TRANSIENT_RETRY_DELAYS[transient_attempt]
                    transient_attempt += 1
                    self.trace.event(
                        "anthropic_transient_retry",
                        {
                            "role": self.role,
                            "attempt": transient_attempt,
                            "delay_seconds": delay,
                            "status_code": exc.status_code,
                        },
                    )
                    time.sleep(delay)
        except ClaudeStreamParseError as exc:
            if self.provider == "anthropic":
                raise ClaudeStopError(
                    "Direct Anthropic stream could not be parsed; exact usage "
                    "is unavailable, so the request cannot be retried safely.",
                    billed_response_count=1,
                ) from exc
            if not exc.request_id:
                _mark_billed_exception(exc, 1)
                raise
            try:
                record = self.cost_tracker.record_generation(exc.request_id)
            except Exception as accounting_error:
                _mark_billed_exception(accounting_error, 1)
                raise
            self.trace.event(
                "malformed_tool_response",
                {
                    "role": self.role,
                    "generation": asdict(record),
                    "budget": self.cost_tracker.snapshot(),
                    "error": str(exc),
                },
            )
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous tool arguments could not be parsed. "
                        "Retry the same action with compact valid JSON. Keep "
                        "report text concise and put each unresolved item in "
                        "the schema's unresolved array."
                    ),
                }
            )
            response = {
                "output": [],
                "stop_reason": "malformed_tool_arguments",
                "billed_response_count": 1,
                "continue_immediately": True,
                "synthetic_recovery": True,
            }
            self.trace.model_call(_jsonable(request), response)
            return response
        self.messages.append(
            {
                "role": "assistant",
                "content": _claude_replay_content(
                    turn.message.content,
                    direct=self.provider == "anthropic",
                ),
            }
        )
        response = self._normalize_response(turn)
        response["billed_response_count"] = 1
        self.trace.model_call(_jsonable(request), response)

        try:
            if self.provider == "anthropic":
                assert isinstance(self.cost_tracker, AnthropicCostTracker)
                record = self.cost_tracker.record_message(turn.message)
                event_name = "anthropic_cost"
                response["anthropic_cost"] = asdict(record)
                response["anthropic_budget"] = self.cost_tracker.snapshot()
            else:
                generation_id = turn.request_id or getattr(turn.message, "id", None)
                record = self.cost_tracker.record_generation(str(generation_id or ""))
                event_name = "openrouter_cost"
                response["openrouter_cost"] = asdict(record)
                response["openrouter_budget"] = self.cost_tracker.snapshot()
        except Exception as exc:
            _mark_billed_exception(exc, 1)
            raise
        self.trace.event(
            event_name,
            {
                "role": self.role,
                "generation": asdict(record),
                "budget": self.cost_tracker.snapshot(),
            },
        )
        if turn.stop_reason == "max_tokens":
            raise ClaudeStopError(
                "Claude reached max_tokens before completing the turn; usage "
                "was recorded",
                billed_response_count=1,
            )
        if turn.stop_reason == "model_context_window_exceeded":
            raise ClaudeStopError(
                "Claude exhausted the context window; usage was recorded",
                billed_response_count=1,
            )
        if turn.stop_reason == "refusal":
            details = getattr(turn.message, "stop_details", None)
            recovery = self._policy_feedback_recovery(
                source="refusal",
                category=_anthropic_policy_category(details),
                error_summary=_anthropic_policy_error_summary(details),
                billed_response_count=1,
                rollback_assistant=True,
            )
            if recovery is not None:
                return recovery
            raise ClaudeStopError(
                f"Claude refused the request after usage was recorded: {details}",
                billed_response_count=1,
            )
        if turn.stop_reason == "pause_turn":
            self._consecutive_pause_turns += 1
            if self._consecutive_pause_turns > 4:
                raise ClaudeStopError(
                    "Claude returned pause_turn more than four times; every "
                    "completed response was traced and accounted",
                    billed_response_count=1,
                )
            response["continue_immediately"] = True
            return response
        self._consecutive_pause_turns = 0
        if turn.stop_reason not in {"end_turn", "tool_use"}:
            raise ClaudeStopError(
                f"Claude returned unsupported stop reason {turn.stop_reason!r}; "
                "the completed response was traced and accounted",
                billed_response_count=1,
            )
        return response

    def call_provider_tools(
        self,
        *,
        tools: list[dict[str, Any]],
        max_output_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call Claude with already translated native Messages tool objects."""

        return self.call(
            tools=[],
            max_output_tokens=max_output_tokens,
            tool_choice=tool_choice,
            _provider_tools=tools,
        )

    @staticmethod
    def _normalize_response(
        turn: Any,
    ) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        for block in getattr(turn.message, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "thinking":
                summary = getattr(block, "thinking", "")
                item: dict[str, Any] = {
                    "type": "reasoning",
                    "summary": [{"text": summary}] if summary else [],
                }
                signature = getattr(block, "signature", None)
                if signature:
                    item["encrypted_content"] = signature
                output.append(item)
            elif block_type == "text":
                output.append(
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": getattr(block, "text", ""),
                            }
                        ],
                    }
                )
            elif block_type == "tool_use":
                output.append(
                    {
                        "type": "function_call",
                        "call_id": getattr(block, "id"),
                        "name": getattr(block, "name"),
                        "arguments": json.dumps(
                            getattr(block, "input", {}) or {},
                            ensure_ascii=False,
                        ),
                    }
                )
        return {
            "id": getattr(turn.message, "id", None),
            "model": getattr(turn.message, "model", None),
            "provider": getattr(turn.message, "provider", None),
            "status": "completed",
            "stop_reason": turn.stop_reason,
            "output": output,
            "usage": _jsonable(getattr(turn.message, "usage", None)),
            "raw_message": turn.message.model_dump(mode="json"),
        }


def _mark_billed_exception(exc: Exception, count: int) -> None:
    prior = getattr(exc, "billed_response_count", 0)
    if isinstance(prior, bool) or not isinstance(prior, int) or prior < 0:
        prior = 0
    try:
        setattr(exc, "billed_response_count", prior + count)
        setattr(exc, "billed_response", True)
    except Exception as metadata_error:
        raise ClaudeStopError(
            f"{type(exc).__name__}: {exc}",
            billed_response_count=prior + count,
        ) from metadata_error
