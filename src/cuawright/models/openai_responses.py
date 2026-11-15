from __future__ import annotations

import base64
import copy
import random
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import requests
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from cuawright.artifacts import TraceWriter
from cuawright.models.base import (
    RUN_COMMAND_DESCRIPTION,
    canonical_tools,
    command_result_metadata,
    command_result_text,
    normalized_turn,
)
from cuawright.models.base import ModelTurn, ToolCall
from cuawright.models.providers import PROVIDER_DEFAULT_BASE_URLS
from cuawright.runtime.contracts import CommandResult, ImageAttachment


PHYAGI_BASE_URL = PROVIDER_DEFAULT_BASE_URLS["openai"]
RESPONSE_RETRY_DELAYS = (2, 4, 8, 16, 30, 45, 60, 90, 120)
POLICY_RETRY_DELAYS = (1, 2, 4, 8, 15, 30, 30, 30, 30)


class ResponsesThread(Protocol):
    def call(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def add_tool_output(self, call_id: str, output: Any) -> None: ...

    def add_image(self, path: str, mime: str, data: bytes) -> None: ...

    def add_user_text(self, text: str) -> None: ...


def create_responses_client(
    *,
    api_key: str,
    base_url: str = PHYAGI_BASE_URL,
) -> OpenAI:
    """Construct the Responses client while leaving retries to the thread."""

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
    )


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


def _normalize_output_items(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in response.get("output", []):
        item_type = item.get("type")
        if item_type == "reasoning":
            reasoning = {
                "type": "reasoning",
                "summary": item.get("summary") or [],
            }
            if item.get("id"):
                reasoning["id"] = item["id"]
            if item.get("encrypted_content"):
                reasoning["encrypted_content"] = item["encrypted_content"]
            normalized.append(reasoning)
        elif item_type == "function_call":
            normalized.append(
                {
                    "type": "function_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item.get("arguments", "{}"),
                }
            )
        elif item_type == "message":
            text = "\n".join(
                content.get("text", "")
                for content in item.get("content") or []
                if content.get("type") == "output_text"
            )
            if text:
                normalized.append({"role": "assistant", "content": text})
    return normalized


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _exception_diagnostics(exc: BaseException) -> str:
    return " <- ".join(
        f"{type(item).__name__}: {item}" for item in _exception_chain(exc)
    )[:4_000]


def _is_transient_response_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            requests.ConnectionError,
            requests.Timeout,
        ),
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 429} or 500 <= exc.status_code < 600
    text = str(exc).lower()
    if re.search(r"\b5\d\d\b", text):
        return True
    transient_markers = (
        "429",
        "rate limit",
        "rate_limit",
        "408",
        "500",
        "502",
        "503",
        "504",
        "520",
        "521",
        "522",
        "523",
        "524",
        "525",
        "526",
        "527",
        "529",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection error",
        "remote end closed",
        "no route to host",
        "name or service not known",
        "temporary failure in name resolution",
        "temporarily unavailable",
        "upstream.failed",
    )
    if any(marker in text for marker in transient_markers):
        return True
    return any(
        any(marker in str(item).lower() for marker in transient_markers)
        for item in _exception_chain(exc)[1:]
    )


def _is_policy_rejection(exc: Exception) -> bool:
    markers = (
        "content_policy_violation",
        "cyber_policy",
        "'code': 'invalid_prompt'",
        '"code": "invalid_prompt"',
        "flagged for possible cybersecurity risk",
        "prompt was flagged as potentially violating our usage policy",
    )
    return any(
        any(marker in str(item).lower() for marker in markers)
        for item in _exception_chain(exc)
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass
    match = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s*seconds?", str(exc), re.I)
    if match:
        return float(match.group(1))
    return None


def _exception_billed_response_count(exc: Exception) -> int:
    count = getattr(exc, "billed_response_count", None)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return 1 if getattr(exc, "billed_response", False) else 0


def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return (
        status_code
        if isinstance(status_code, int) and not isinstance(status_code, bool)
        else None
    )


def _policy_category(exc: Exception) -> str:
    text = " ".join(str(item).lower() for item in _exception_chain(exc))
    if "cyber" in text:
        return "cyber_policy"
    if "content_policy_violation" in text or "content policy" in text:
        return "content_policy"
    return "policy"


def _policy_feedback(
    *, attempt: int, maximum: int, category: str, error_summary: str
) -> str:
    return (
        "The provider rejected the immediately preceding request with this "
        f"policy error (compliance retry {attempt}/{maximum}, category "
        f"{category}): {error_summary[:1_000]} Do not repeat, disguise, or work "
        "around the blocked action. Reassess the OSWorld task and continue only "
        "with a clearly benign, policy-compliant approach using the existing "
        "tools. Prefer ordinary user-visible application operations. Avoid "
        "security-sensitive commands or techniques. If an image caused the "
        "block, inspect the VM with narrower benign screenshots, OCR, or metadata "
        "instead of attempting to bypass the safeguard. If no compliant "
        "continuation is possible, stop rather than evading the policy."
    )


class ResponsesThread:
    """PhyAGI/OpenAI Responses conversation with exact retry semantics."""

    def __init__(
        self,
        *,
        client: OpenAI,
        trace: TraceWriter,
        model: str,
        reasoning_effort: str,
        role: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        base_url: str,
        infrastructure: str | None = "aiservices",
        policy_feedback_max_retries: int = 0,
    ) -> None:
        if (
            isinstance(policy_feedback_max_retries, bool)
            or not isinstance(policy_feedback_max_retries, int)
            or policy_feedback_max_retries < 0
        ):
            raise ValueError("policy_feedback_max_retries must be nonnegative")
        self.client = client
        self.trace = trace
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_output_tokens = max_output_tokens
        self.session_id = f"osworld-cli-{role}-{uuid.uuid4().hex}"
        self.context: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.base_url = base_url
        self.infrastructure = infrastructure
        self.policy_feedback_max_retries = policy_feedback_max_retries

    def add_user_text(self, text: str) -> None:
        self.context.append({"role": "user", "content": text})

    def add_tool_output(self, call_id: str, output: Any) -> None:
        self.context.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        )

    def add_tool_outputs(self, outputs: list[tuple[str, Any]]) -> None:
        for call_id, output in outputs:
            self.add_tool_output(call_id, output)

    def add_image(self, path: str, mime: str, data: bytes) -> None:
        encoded = base64.b64encode(data).decode("ascii")
        self.context.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Image read from VM path: {path}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{encoded}",
                        "detail": "auto",
                    },
                ],
            }
        )

    def _drop_blocked_images(self) -> int:
        removed = 0
        normalized: list[dict[str, Any]] = []
        for item in self.context:
            if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                normalized.append(item)
                continue
            content = []
            item_removed = 0
            for part in item["content"]:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    removed += 1
                    item_removed += 1
                    continue
                content.append(part)
            if item_removed and not content:
                content = [
                    {
                        "type": "input_text",
                        "text": "An image attachment was unavailable to the model.",
                    }
                ]
            normalized.append({**item, "content": content})
        if removed:
            self.context = normalized
            self.context.append(
                {
                    "role": "user",
                    "content": (
                        "The image attachment was blocked by the model endpoint. "
                        "Continue using VM-local inspection tools, OCR, metadata, "
                        "or screenshots of narrower non-sensitive regions."
                    ),
                }
            )
        return removed

    def call(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_tools = self.tools if tools is None else tools
        extra_body: dict[str, Any] = {
            "tier": "base",
            "session_id": self.session_id,
            "strict_session": True,
        }
        if self.infrastructure is not None:
            extra_body["infrastructure"] = self.infrastructure
        request: dict[str, Any] = {
            "model": self.model,
            "input": self.context,
            "tools": active_tools,
            "parallel_tool_calls": False,
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": "auto",
            },
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "max_output_tokens": max_output_tokens or self.max_output_tokens,
            "extra_body": extra_body,
        }
        if active_tools and tool_choice is not None:
            request["tool_choice"] = tool_choice

        frozen_request = copy.deepcopy(request)
        last_error: Exception | None = None
        billed_response_count = 0
        transient_attempt = 0
        unchanged_policy_attempt = 0
        policy_feedback_attempt = 0
        while True:
            try:
                sent_request = copy.deepcopy(frozen_request)
                response_obj = self.client.responses.create(**sent_request)
                response = _jsonable(response_obj)
                if billed_response_count:
                    response_billed = response.get("billed_response_count", 1)
                    if (
                        isinstance(response_billed, bool)
                        or not isinstance(response_billed, int)
                        or response_billed < 0
                    ):
                        response_billed = 1
                    response["billed_response_count"] = (
                        response_billed + billed_response_count
                    )
                self.trace.model_call(sent_request, response)
                self.context.extend(_normalize_output_items(response))
                return response
            except Exception as exc:
                last_error = exc
                billed_response_count += _exception_billed_response_count(exc)
                if _is_policy_rejection(exc):
                    if self.policy_feedback_max_retries:
                        if policy_feedback_attempt >= self.policy_feedback_max_retries:
                            self.trace.event(
                                "policy_feedback_retries_exhausted",
                                {
                                    "role": self.role,
                                    "attempt": policy_feedback_attempt,
                                    "max_attempts": self.policy_feedback_max_retries,
                                    "category": _policy_category(exc),
                                    "exception_type": type(exc).__name__,
                                    "status_code": _exception_status_code(exc),
                                    "billed_response_count": billed_response_count,
                                    "error": _exception_diagnostics(exc),
                                },
                            )
                            break
                        policy_feedback_attempt += 1
                        feedback = _policy_feedback(
                            attempt=policy_feedback_attempt,
                            maximum=self.policy_feedback_max_retries,
                            category=_policy_category(exc),
                            error_summary=_exception_diagnostics(exc),
                        )
                        self.context.append({"role": "user", "content": feedback})
                        frozen_request["input"] = copy.deepcopy(self.context)
                        self.trace.event(
                            "policy_feedback_retry",
                            {
                                "role": self.role,
                                "attempt": policy_feedback_attempt,
                                "max_attempts": self.policy_feedback_max_retries,
                                "category": _policy_category(exc),
                                "status_code": _exception_status_code(exc),
                                "billed_response_count": billed_response_count,
                                "error": _exception_diagnostics(exc),
                            },
                        )
                        continue
                    if unchanged_policy_attempt == 9:
                        self.trace.event(
                            "policy_retries_exhausted",
                            {
                                "role": self.role,
                                "attempt": unchanged_policy_attempt + 1,
                                "max_attempts": 10,
                                "exception_type": type(exc).__name__,
                                "status_code": _exception_status_code(exc),
                                "billed_response_count": billed_response_count,
                                "error": _exception_diagnostics(exc),
                            },
                        )
                        break
                    base_delay = POLICY_RETRY_DELAYS[unchanged_policy_attempt]
                    retry_after = _retry_after_seconds(exc) or 0.0
                    delay = max(base_delay, retry_after) + random.uniform(0.0, 3.0)
                    unchanged_policy_attempt += 1
                    self.trace.event(
                        "policy_retry",
                        {
                            "role": self.role,
                            "attempt": unchanged_policy_attempt,
                            "max_attempts": 10,
                            "delay_seconds": round(delay, 3),
                            "exception_type": type(exc).__name__,
                            "status_code": _exception_status_code(exc),
                            "billed_response_count": billed_response_count,
                            "error": _exception_diagnostics(exc),
                        },
                    )
                    time.sleep(delay)
                    continue
                if not _is_transient_response_error(exc):
                    break
                if transient_attempt == 9:
                    break
                base_delay = RESPONSE_RETRY_DELAYS[transient_attempt]
                retry_after = _retry_after_seconds(exc) or 0.0
                delay = max(base_delay, retry_after) + random.uniform(0.0, 3.0)
                transient_attempt += 1
                self.trace.event(
                    "response_retry",
                    {
                        "role": self.role,
                        "attempt": transient_attempt,
                        "max_attempts": 10,
                        "delay_seconds": round(delay, 3),
                        "exception_type": type(exc).__name__,
                        "error": _exception_diagnostics(exc),
                    },
                )
                time.sleep(delay)
        if last_error is None:
            raise RuntimeError("Responses API call failed without an exception")
        error = RuntimeError(
            f"Responses API call failed: {_exception_diagnostics(last_error)}"
        )
        error.billed_response_count = billed_response_count
        error.billed_response = billed_response_count > 0
        error.status_code = _exception_status_code(last_error)
        raise error from last_error


def openai_responses_tools(
    tools: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Translate the frozen core schema to Responses function-tool form."""

    canonical = canonical_tools(tools)
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": RUN_COMMAND_DESCRIPTION,
            "parameters": copy.deepcopy(tool["parameters"]),
            "strict": tool.get("strict") is True,
        }
        for tool in canonical
    ]


class OpenAIResponsesAdapter:
    """Compatibility adapter around the established Responses thread."""

    def __init__(self, thread: ResponsesThread) -> None:
        self.thread = thread

    def next_turn(
        self,
        *,
        tools: Sequence[Mapping[str, object]],
    ) -> ModelTurn:
        response = self.thread.call(tools=openai_responses_tools(tools))
        if not isinstance(response, Mapping):
            raise TypeError("ResponsesThread.call() must return a mapping")
        return normalized_turn(response, first_tool_call_only=True)

    def add_command_result(
        self,
        tool_call: ToolCall,
        result: CommandResult,
    ) -> None:
        metadata = command_result_metadata(result)
        images = [
            attachment
            for attachment in result.attachments
            if isinstance(attachment, ImageAttachment)
        ]
        self.thread.add_tool_output(
            tool_call.call_id,
            command_result_text(metadata, model_output=result.model_output),
        )
        for image in images:
            self.thread.add_image(
                image.source_path or "<terminal-image>",
                image.media_type,
                image.data,
            )

    def add_user_text(self, text: str) -> None:
        self.thread.add_user_text(text)
