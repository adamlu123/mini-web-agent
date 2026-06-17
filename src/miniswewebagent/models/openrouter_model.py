"""OpenRouter model backend.

OpenRouter (https://openrouter.ai) exposes a standard OpenAI-compatible
`/chat/completions` API, which differs from the `/responses` API used by
`PhyagiModel`. This class reuses PhyagiModel's parsing / observation
formatting / retry machinery and overrides only the HTTP request and
payload/response serialization bits.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, field_validator

from miniswewebagent.models.phyagi_model import (
    DEFAULT_OBSERVATION_TEMPLATE,
    DEFAULT_XML_FORMAT_ERROR_TEMPLATE,
    MAX_JSON_PARSE_RETRIES,
    MAX_RATE_LIMIT_RETRIES,
    MAX_TRANSIENT_GATEWAY_RETRIES,
    PhyagiModel,
    _is_rate_limit_error,
    _is_transient_gateway_error,
    _request_metrics_from_serialized_input,
    _safe_int,
    _validate_bash_command,
)
from miniswewebagent.utils.logging import append_runtime_log


def _serialize_chat_content_part(part: dict[str, Any]) -> dict[str, Any]:
    if part.get("type") == "input_image":
        return {"type": "image_url", "image_url": {"url": part.get("image_url", "")}}
    return {"type": "text", "text": part.get("text", "")}


def _serialize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert harness messages to OpenAI chat-completions format."""
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "exit":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
            continue
        parts = [_serialize_chat_content_part(p) for p in content if isinstance(p, dict)]
        # If all parts are plain text, collapse to a string (max model compat).
        if parts and all(p.get("type") == "text" for p in parts):
            serialized.append({"role": role, "content": "\n".join(p["text"] for p in parts)})
        else:
            serialized.append({"role": role, "content": parts})
    return serialized


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _usage_from_chat_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": _safe_int(usage.get("prompt_tokens")),
        "output_tokens": _safe_int(usage.get("completion_tokens")),
        "total_tokens": _safe_int(usage.get("total_tokens")),
        "cached_input_tokens": _safe_int(details.get("cached_tokens")),
        "reasoning_output_tokens": _safe_int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        ),
    }


class OpenRouterModelConfig(BaseModel):
    model_name: str = "qwen/qwen3.5-27b"
    openrouter_api_key: str = ""
    openrouter_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_referer: str = "https://github.com/adamlu123/mini-web-agent"
    openrouter_app_title: str = "mini-web-agent"
    max_output_tokens: int = 4000
    request_timeout_seconds: int = 120
    error_log_path: Path | None = None
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    response_mode: str = "xml"
    format_error_template: str = DEFAULT_XML_FORMAT_ERROR_TEMPLATE
    attach_observation_screenshot: bool = True

    @field_validator(
        "model_name",
        "openrouter_api_key",
        "openrouter_endpoint",
        "openrouter_referer",
        "openrouter_app_title",
        "observation_template",
        "response_mode",
        "format_error_template",
        mode="before",
    )
    @classmethod
    def _normalize_optional_strings(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class OpenRouterModel(PhyagiModel):
    """Model backend talking to the OpenRouter chat-completions endpoint."""

    def _ensure_auth(self) -> None:
        if not self.config.openrouter_api_key:
            self.config.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.config.openrouter_api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY.")

    # Intentionally bypass PhyagiModel.__init__ — it validates
    # OPENAI_GATEWAY_API_KEY, which doesn't apply here. We re-establish the
    # bookkeeping it sets up.
    def __init__(self, *, config_class: type = OpenRouterModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        zero_req = {k: 0 for k in ("message_count", "text_part_count", "image_part_count", "text_chars", "serialized_chars")}
        zero_use = {k: 0 for k in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_output_tokens")}
        self._last_request_metrics = dict(zero_req)
        self._last_usage_metrics = dict(zero_use)
        self._cumulative_request_metrics = dict(zero_req)
        self._cumulative_usage_metrics = dict(zero_use)
        self._ensure_auth()

    def _log_gateway_error(self, *, event: str, attempt: int, error: BaseException) -> None:
        # Override PhyagiModel._log_gateway_error which references
        # self.config.openai_gateway_endpoint (not present on
        # OpenRouterModelConfig). Use openrouter_endpoint instead.
        from miniswewebagent.utils.logging import append_runtime_log

        response = getattr(error, "response", None)
        response_text = ""
        if response is not None:
            try:
                response_text = str(getattr(response, "text", "") or "")
            except Exception:
                response_text = ""
        if len(response_text) > 4000:
            response_text = response_text[:4000]

        append_runtime_log(
            self.config.error_log_path,
            source="openrouter",
            event=event,
            model_name=self.config.model_name,
            endpoint=self.config.openrouter_endpoint,
            attempt=attempt,
            error_type=type(error).__name__,
            error=str(error),
            status_code=getattr(response, "status_code", None),
            response_text=response_text,
        )

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "messages": _serialize_chat_messages(messages),
            "max_tokens": self.config.max_output_tokens,
        }

    async def _query_async(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "HTTP-Referer": self.config.openrouter_referer,
            "X-Title": self.config.openrouter_app_title,
        }

        last_error: ValueError | None = None
        raw_text = ""
        request_messages = list(messages)
        for attempt_index in range(MAX_JSON_PARSE_RETRIES + 1):
            payload = self._build_payload(request_messages)
            request_metrics = _request_metrics_from_serialized_input(payload["messages"])
            self._last_request_metrics = dict(request_metrics)
            for key, value in request_metrics.items():
                self._cumulative_request_metrics[key] += value

            response_payload = None
            for rate_limit_attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
                        response = await client.post(
                            self.config.openrouter_endpoint,
                            headers=headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        response_payload = response.json()
                    break
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        self._log_gateway_error(event="rate_limit_error", attempt=rate_limit_attempt + 1, error=exc)
                        if rate_limit_attempt >= MAX_RATE_LIMIT_RETRIES:
                            raise
                        await asyncio.sleep(min(5 * (rate_limit_attempt + 1), 30))
                        continue
                    if _is_transient_gateway_error(exc):
                        self._log_gateway_error(event="transient_gateway_error", attempt=rate_limit_attempt + 1, error=exc)
                        if rate_limit_attempt >= MAX_TRANSIENT_GATEWAY_RETRIES:
                            raise
                        await asyncio.sleep(min(2 * (rate_limit_attempt + 1), 10))
                        continue
                    self._log_gateway_error(event="fatal_gateway_error", attempt=rate_limit_attempt + 1, error=exc)
                    raise

            if response_payload is None:
                raise RuntimeError("OpenRouter request returned no payload.")

            usage_metrics = _usage_from_chat_payload(response_payload)
            self._last_usage_metrics = dict(usage_metrics)
            for key, value in usage_metrics.items():
                self._cumulative_usage_metrics[key] += value

            raw_text = _extract_chat_text(response_payload)
            append_runtime_log(
                self._raw_response_log_path(),
                source="model",
                event="raw_text",
                attempt=attempt_index + 1,
                raw_text=raw_text,
            )
            try:
                parsed = self._parse_model_output(raw_text)
                break
            except ValueError as exc:
                last_error = exc
                if attempt_index < MAX_JSON_PARSE_RETRIES:
                    request_messages.append(self._format_repair_message(raw_text=raw_text, error=str(exc)))
        else:
            raise self._format_error(
                raw_text=raw_text,
                error=str(last_error or ValueError("Unable to parse model output.")),
            )

        actions = []
        bash_command = parsed.get("bash_command", "").strip()
        python_code = parsed.get("python_code", "").strip()
        if bash_command:
            try:
                _validate_bash_command(bash_command)
            except ValueError as exc:
                raise self._format_error(raw_text=raw_text, error=str(exc))
            actions.append({"bash_command": bash_command, "command": bash_command})
        elif python_code:
            actions.append({"python_code": python_code})

        return self.format_message(
            role="assistant",
            content=parsed.get("thought", ""),
            extra={
                "actions": actions,
                "done": bool(parsed.get("done", False)),
                "final_response": parsed.get("final_response", ""),
                "raw_response": parsed,
                "usage": self._usage_snapshot(),
            },
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "model": {
                "config": {
                    **self.config.model_dump(mode="json"),
                    "openrouter_api_key": "<redacted>",
                },
                "usage": {**self._usage_snapshot()},
                "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            }
        }
