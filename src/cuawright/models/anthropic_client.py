from __future__ import annotations

import inspect
import json
import os
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from anthropic import AsyncAnthropic, DefaultAioHttpClient

CLAUDE_OPUS_5 = "claude-opus-5"
Effort = Literal["low", "medium", "high", "xhigh", "max"]
ThinkingDisplay = Literal["summarized", "omitted"]
ThinkingPrefixMismatchBehavior = Literal["drop_block"]
ToolHandler = Callable[[dict[str, Any]], str | Awaitable[str]]


class ClaudeClientError(RuntimeError):
    """Base error for deterministic client-side Claude failures."""


class ClaudeStopError(ClaudeClientError):
    """Raised when Claude stops without a complete answer or actionable tools."""

    def __init__(
        self,
        message: str,
        *,
        billed_response: bool = False,
        billed_response_count: int | None = None,
    ) -> None:
        super().__init__(message)
        if billed_response_count is None:
            billed_response_count = 1 if billed_response else 0
        if (
            isinstance(billed_response_count, bool)
            or not isinstance(billed_response_count, int)
            or billed_response_count < 0
        ):
            raise ValueError("billed_response_count must be a non-negative integer")
        self.billed_response_count = billed_response_count
        self.billed_response = billed_response_count > 0


class ClaudeStreamParseError(ClaudeClientError):
    """Raised when a streamed response cannot assemble tool argument JSON."""

    def __init__(self, message: str, *, request_id: str | None) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.billed_response_count = 0
        self.billed_response = False


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _usage_rows(usage: Any) -> list[Any]:
    iterations = list(_field(usage, "iterations") or [])
    return iterations or [usage]


@dataclass(frozen=True)
class ClaudeClientConfig:
    model: str = CLAUDE_OPUS_5
    max_tokens: int = 64_000
    effort: Effort = "max"
    thinking_display: ThinkingDisplay = "summarized"
    cache_ttl: Literal["5m", "1h"] = "5m"
    server_compaction: bool = True
    thinking_prefix_mismatch_behavior: ThinkingPrefixMismatchBehavior | None = None
    compaction_trigger_tokens: int = 150_000
    disable_parallel_tool_use: bool = True
    max_retries: int = 4
    policy_feedback_max_retries: int = 0
    timeout_seconds: float = 1_200.0
    max_tool_rounds: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if type(self.policy_feedback_max_retries) is not int:
            raise TypeError("policy_feedback_max_retries must be an integer")
        if self.policy_feedback_max_retries < 0:
            raise ValueError("policy_feedback_max_retries cannot be negative")
        if self.compaction_trigger_tokens < 50_000:
            raise ValueError("compaction_trigger_tokens must be at least 50,000")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be positive")

    @property
    def cache_control(self) -> dict[str, str]:
        value = {"type": "ephemeral"}
        if self.cache_ttl == "1h":
            value["ttl"] = "1h"
        return value

    @property
    def context_management(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {
                        "type": "input_tokens",
                        "value": self.compaction_trigger_tokens,
                    },
                }
            ]
        }

    @property
    def betas(self) -> list[str]:
        values = []
        if self.server_compaction:
            values.append("compact-2026-01-12")
        if self.thinking_prefix_mismatch_behavior is not None:
            values.append("thinking-binding-controls-2026-08-01")
        return values


@dataclass
class ClaudeUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    thinking_tokens: int = 0

    def add_message(self, message: Any) -> None:
        usage = _field(message, "usage")
        if usage is None:
            return
        self.requests += 1
        iteration_thinking = 0
        for row in _usage_rows(usage):
            self.input_tokens += int(_field(row, "input_tokens") or 0)
            self.output_tokens += int(_field(row, "output_tokens") or 0)
            self.cache_creation_input_tokens += int(
                _field(row, "cache_creation_input_tokens") or 0
            )
            self.cache_read_input_tokens += int(
                _field(row, "cache_read_input_tokens") or 0
            )
            details = _field(row, "output_tokens_details")
            iteration_thinking += int(_field(details, "thinking_tokens") or 0)
        if iteration_thinking:
            self.thinking_tokens += iteration_thinking
        else:
            details = _field(usage, "output_tokens_details")
            self.thinking_tokens += int(_field(details, "thinking_tokens") or 0)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class ClaudeTurn:
    message: Any
    request_id: str | None
    stop_reason: str | None
    text: str
    thinking_summaries: list[str]
    tool_uses: list[Any]

    @classmethod
    def from_message(
        cls,
        message: Any,
        *,
        request_id: str | None = None,
    ) -> ClaudeTurn:
        text: list[str] = []
        thinking: list[str] = []
        tools: list[Any] = []
        for block in getattr(message, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text.append(getattr(block, "text", ""))
            elif block_type == "thinking":
                summary = getattr(block, "thinking", "")
                if summary:
                    thinking.append(summary)
            elif block_type == "tool_use":
                tools.append(block)
        return cls(
            message=message,
            request_id=request_id or getattr(message, "_request_id", None),
            stop_reason=getattr(message, "stop_reason", None),
            text="".join(text),
            thinking_summaries=thinking,
            tool_uses=tools,
        )

    def public_summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "stop_reason": self.stop_reason,
            "text": self.text,
            "thinking_summaries": self.thinking_summaries,
            "tool_uses": [
                {
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", None),
                }
                for block in self.tool_uses
            ],
        }


@dataclass
class ClaudeRunResult:
    final_turn: ClaudeTurn
    turns: list[ClaudeTurn]
    messages: list[dict[str, Any]]
    usage: ClaudeUsage = field(default_factory=ClaudeUsage)

    def public_summary(self) -> dict[str, Any]:
        return {
            "model": getattr(self.final_turn.message, "model", None),
            "final": self.final_turn.public_summary(),
            "turns": [turn.public_summary() for turn in self.turns],
            "usage": self.usage.to_dict(),
        }


def read_api_key(path: str | os.PathLike[str]) -> str:
    key_path = Path(path).expanduser()
    if not key_path.is_file():
        raise FileNotFoundError(f"Claude API key file does not exist: {key_path}")
    mode = stat.S_IMODE(key_path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"Claude API key file must not be accessible by group/others: "
            f"{key_path} has mode {mode:04o}"
        )
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"Claude API key file is empty: {key_path}")
    return key


class ClaudeMessagesClient:
    """First-class Claude Messages client for long-running tool agents.

    The client uses Anthropic's official async SDK and streaming helper,
    adaptive thinking with encrypted signature round-tripping, automatic
    prompt caching, SDK retries, and explicit stop-reason handling.
    """

    def __init__(
        self,
        *,
        api_key: str,
        config: ClaudeClientConfig | None = None,
        client: AsyncAnthropic | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or ClaudeClientConfig()
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            client_options: dict[str, Any] = {
                "api_key": api_key,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout_seconds,
                "http_client": DefaultAioHttpClient(),
            }
            if base_url is not None:
                client_options["base_url"] = base_url
            if default_headers is not None:
                client_options["default_headers"] = dict(default_headers)
            self._client = AsyncAnthropic(**client_options)

    @classmethod
    def from_key_file(
        cls,
        path: str | os.PathLike[str] = "~/.claude-key",
        *,
        config: ClaudeClientConfig | None = None,
    ) -> ClaudeMessagesClient:
        return cls(api_key=read_api_key(path), config=config)

    async def __aenter__(self) -> ClaudeMessagesClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def count_tokens(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        system: str | Sequence[dict[str, Any]] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "thinking": {
                "type": "adaptive",
                "display": self.config.thinking_display,
            },
            "output_config": {"effort": self.config.effort},
            "cache_control": self.config.cache_control,
        }
        if system is not None:
            request["system"] = system
        if tools:
            request["tools"] = list(tools)
        if self.config.betas:
            request["betas"] = self.config.betas
        if self.config.server_compaction:
            request["context_management"] = self.config.context_management
        if self.config.thinking_prefix_mismatch_behavior is not None:
            request["thinking"]["block_binding"] = {
                "prefix_mismatch_behavior": (
                    self.config.thinking_prefix_mismatch_behavior
                )
            }
        if self.config.betas:
            response = await self._client.beta.messages.count_tokens(**request)
        else:
            response = await self._client.messages.count_tokens(**request)
        return int(response.input_tokens)

    async def create_turn(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        system: str | Sequence[dict[str, Any]] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ClaudeTurn:
        request = build_create_request(
            self.config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        return await self.create_prepared_turn(request=request)

    async def create_prepared_turn(
        self,
        *,
        request: Mapping[str, Any],
    ) -> ClaudeTurn:
        """Send an already built request without reconstructing its payload."""
        # Streaming avoids idle-connection failures on long, high-effort turns.
        messages_api = (
            self._client.beta.messages if self.config.betas else self._client.messages
        )
        async with messages_api.stream(**dict(request)) as stream:
            try:
                message = await stream.get_final_message()
            except ValueError as exc:
                raise ClaudeStreamParseError(
                    str(exc),
                    request_id=stream.request_id,
                ) from exc
            request_id = stream.request_id
        return ClaudeTurn.from_message(message, request_id=request_id)

    async def run_tool_loop(
        self,
        *,
        user_prompt: str,
        tools: Sequence[dict[str, Any]],
        tool_handlers: Mapping[str, ToolHandler],
        system: str | Sequence[dict[str, Any]] | None = None,
    ) -> ClaudeRunResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        usage = ClaudeUsage()
        turns: list[ClaudeTurn] = []

        for _ in range(self.config.max_tool_rounds):
            turn = await self.create_turn(
                messages=messages,
                system=system,
                tools=tools,
                tool_choice={
                    "type": "auto",
                    "disable_parallel_tool_use": (
                        self.config.disable_parallel_tool_use
                    ),
                },
            )
            turns.append(turn)
            usage.add_message(turn.message)

            # Preserve every assistant content block unchanged. This includes
            # thinking summaries and encrypted signatures required for
            # multi-turn and interleaved-thinking continuity.
            messages.append({"role": "assistant", "content": turn.message.content})

            if turn.stop_reason == "tool_use":
                if not turn.tool_uses:
                    raise ClaudeStopError(
                        "Claude returned stop_reason=tool_use without a tool_use block"
                    )
                tool_results = []
                for tool_use in turn.tool_uses:
                    name = getattr(tool_use, "name", "")
                    handler = tool_handlers.get(name)
                    if handler is None:
                        result = f"No handler is registered for tool {name!r}."
                        is_error = True
                    else:
                        try:
                            value = handler(dict(getattr(tool_use, "input", {}) or {}))
                            if inspect.isawaitable(value):
                                value = await value
                            result = str(value)
                            is_error = False
                        except Exception as exc:
                            result = f"{type(exc).__name__}: {exc}"
                            is_error = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result,
                            "is_error": is_error,
                        }
                    )

                # Anthropic recommends sending tool_result blocks alone without
                # appending explanatory text after them.
                messages.append({"role": "user", "content": tool_results})
                continue

            if turn.stop_reason == "pause_turn":
                # Server-tool loops can pause after their internal iteration
                # limit. Sending the complete assistant content resumes them.
                continue

            if turn.stop_reason == "end_turn":
                return ClaudeRunResult(
                    final_turn=turn,
                    turns=turns,
                    messages=messages,
                    usage=usage,
                )

            if turn.stop_reason == "max_tokens":
                raise ClaudeStopError(
                    "Claude reached max_tokens before completing the turn"
                )
            if turn.stop_reason == "model_context_window_exceeded":
                raise ClaudeStopError("Claude exhausted the model context window")
            if turn.stop_reason == "refusal":
                details = getattr(turn.message, "stop_details", None)
                raise ClaudeStopError(f"Claude refused the request: {details}")
            raise ClaudeStopError(f"Unhandled Claude stop reason: {turn.stop_reason!r}")

        raise ClaudeStopError(
            f"Claude exceeded max_tool_rounds={self.config.max_tool_rounds}"
        )


def dump_private_trace(result: ClaudeRunResult, path: Path) -> None:
    """Persist complete messages, including encrypted thinking signatures.

    This artifact is needed to resume a tool conversation faithfully. Treat it
    as private operational state; it may contain prompts, tool inputs/results,
    and encrypted thinking signatures.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": result.final_turn.message.model,
        "usage": result.usage.to_dict(),
        "messages": [
            {
                "role": message["role"],
                "content": _jsonable(message["content"]),
            }
            for message in result.messages
        ],
    }
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def build_create_request(
    config: ClaudeClientConfig,
    *,
    messages: Sequence[dict[str, Any]],
    system: str | Sequence[dict[str, Any]] | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact SDK arguments used for a streamed Messages request."""
    request: dict[str, Any] = {
        "model": config.model,
        "max_tokens": max_tokens or config.max_tokens,
        "messages": list(messages),
        "thinking": {
            "type": "adaptive",
            "display": config.thinking_display,
        },
        "output_config": {"effort": config.effort},
        "cache_control": config.cache_control,
    }
    if system is not None:
        request["system"] = system
    if tools:
        request["tools"] = list(tools)
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    if config.thinking_prefix_mismatch_behavior is not None:
        request["thinking"]["block_binding"] = {
            "prefix_mismatch_behavior": config.thinking_prefix_mismatch_behavior
        }
    if config.betas:
        request["betas"] = config.betas
    if config.server_compaction:
        request["context_management"] = config.context_management
    if extra_body:
        request["extra_body"] = dict(extra_body)
    return request


def serialized_request_bytes(request: Mapping[str, Any]) -> int:
    """Return a conservative JSON body size including nested base64 data."""
    payload = json.dumps(
        _jsonable(dict(request)),
        ensure_ascii=True,
        default=str,
    )
    return len(payload.encode("utf-8"))
