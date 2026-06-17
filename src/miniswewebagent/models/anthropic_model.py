"""Anthropic (Claude) model backend.

Talks to the Anthropic Messages API (https://api.anthropic.com/v1/messages).
Reuses PhyagiModel's parsing / observation formatting / retry machinery and
overrides only the HTTP request and payload/response serialization bits.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import random
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, field_validator

from miniswewebagent.models.phyagi_model import (
    DEFAULT_OBSERVATION_TEMPLATE,
    DEFAULT_XML_FORMAT_ERROR_TEMPLATE,
    MAX_JSON_PARSE_RETRIES,
    PhyagiModel,
    _is_rate_limit_error,
    _is_transient_gateway_error,
    _request_metrics_from_serialized_input,
    _safe_int,
    _validate_bash_command,
)
from miniswewebagent.utils.logging import append_runtime_log

# Anthropic-specific retry limits (independent of the phyagi defaults).
# The Claude Opus org-level 2M ITPM cap can be saturated for minutes at high
# concurrency, so we allow many more retries and longer backoffs than phyagi.
MAX_RATE_LIMIT_RETRIES = 50
MAX_TRANSIENT_GATEWAY_RETRIES = 20
# Rate-limit retries: uniform random wait in [MIN, MAX] seconds. Randomization
# desynchronizes workers so they stop marching in lockstep into the TPM cap.
RATE_LIMIT_BACKOFF_MIN_SECONDS = 30.0
RATE_LIMIT_BACKOFF_MAX_SECONDS = 60.0
TRANSIENT_BACKOFF_BASE_SECONDS = 1.5
TRANSIENT_BACKOFF_CAP_SECONDS = 60.0


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract a Retry-After hint (seconds) from an httpx error, if any."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    header = response.headers.get("retry-after") if getattr(response, "headers", None) else None
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except (TypeError, ValueError):
        return None


def _image_source_from_url(image_url: str) -> dict[str, Any]:
    """Convert an image URL / data URI into Anthropic `source` form."""
    if image_url.startswith("data:"):
        header, _, encoded = image_url.partition(",")
        media_type = header.split(";")[0].removeprefix("data:") or "image/png"
        return {"type": "base64", "media_type": media_type, "data": encoded}
    # Remote URLs are supported via the newer "url" source type.
    return {"type": "url", "url": image_url}


def _serialize_anthropic_content_part(part: dict[str, Any]) -> dict[str, Any]:
    if part.get("type") == "input_image":
        return {"type": "image", "source": _image_source_from_url(part.get("image_url", ""))}
    return {"type": "text", "text": part.get("text", "")}


def _serialize_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert harness messages to Anthropic Messages API format.

    Returns (system_prompt, messages). Anthropic puts system content in a
    top-level field rather than in the messages list, and only supports
    alternating user/assistant roles.
    """
    system_chunks: list[str] = []
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "exit":
            continue
        content = message.get("content", "")
        if role == "system":
            if isinstance(content, str):
                if content:
                    system_chunks.append(content)
            else:
                for part in content:
                    if isinstance(part, dict) and part.get("type") != "input_image":
                        text = part.get("text", "")
                        if text:
                            system_chunks.append(text)
            continue

        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
            continue
        parts = [_serialize_anthropic_content_part(p) for p in content if isinstance(p, dict)]
        if parts and all(p.get("type") == "text" for p in parts):
            serialized.append({"role": role, "content": "\n".join(p["text"] for p in parts)})
        else:
            serialized.append({"role": role, "content": parts})

    system_prompt = "\n\n".join(system_chunks) if system_chunks else None
    return system_prompt, serialized


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    content = payload.get("content") or []
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                texts.append(text)
    return "\n".join(texts)


def _usage_from_anthropic_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    input_tokens = _safe_int(usage.get("input_tokens"))
    output_tokens = _safe_int(usage.get("output_tokens"))
    cached_input_tokens = _safe_int(usage.get("cache_read_input_tokens"))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_output_tokens": 0,
    }


def _metrics_input_from_anthropic(
    system_prompt: str | None, anthropic_messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Shape data to match what `_request_metrics_from_serialized_input` expects."""
    items: list[dict[str, Any]] = []
    if system_prompt:
        items.append({"content": [{"type": "input_text", "text": system_prompt}]})
    for msg in anthropic_messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            items.append({"content": [{"type": "input_text", "text": content}]})
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append({"type": "input_text", "text": part.get("text", "")})
            elif part.get("type") == "image":
                parts.append({"type": "input_image"})
        items.append({"content": parts})
    return items


class AnthropicModelConfig(BaseModel):
    model_name: str = "claude-opus-4-7"
    anthropic_api_key: str = ""
    anthropic_endpoint: str = "https://api.anthropic.com/v1/messages"
    anthropic_version: str = "2023-06-01"
    max_output_tokens: int = 8000
    request_timeout_seconds: int = 120
    error_log_path: Path | None = None
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    response_mode: str = "xml"
    format_error_template: str = DEFAULT_XML_FORMAT_ERROR_TEMPLATE
    attach_observation_screenshot: bool = True

    @field_validator(
        "model_name",
        "anthropic_api_key",
        "anthropic_endpoint",
        "anthropic_version",
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


class AnthropicModel(PhyagiModel):
    """Model backend talking to the Anthropic Messages API."""

    _CRED_FILE_FALLBACKS = (
        os.environ.get("MINI_WEB_AGENT_CREDENTIALS_FILE", ""),
        "/home/luyadong/cred.sh",
    )

    @classmethod
    def _read_key_from_cred_file(cls) -> str:
        pattern = re.compile(r"^\s*export\s+ANTHROPIC_API_KEY=(.*)$")
        for path_str in cls._CRED_FILE_FALLBACKS:
            if not path_str:
                continue
            try:
                text = Path(path_str).read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                match = pattern.match(line)
                if not match:
                    continue
                value = match.group(1).strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                if value:
                    return value
        return ""

    def _ensure_auth(self) -> None:
        if not self.config.anthropic_api_key:
            self.config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.config.anthropic_api_key:
            self.config.anthropic_api_key = self._read_key_from_cred_file()
        if not self.config.anthropic_api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY.")

    # Bypass PhyagiModel.__init__: it requires OPENAI_GATEWAY_API_KEY which
    # is unrelated here. Re-establish the bookkeeping state it sets up.
    def __init__(self, *, config_class: type = AnthropicModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        zero_req = {
            k: 0
            for k in (
                "message_count",
                "text_part_count",
                "image_part_count",
                "text_chars",
                "serialized_chars",
            )
        }
        zero_use = {
            k: 0
            for k in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_input_tokens",
                "reasoning_output_tokens",
            )
        }
        self._last_request_metrics = dict(zero_req)
        self._last_usage_metrics = dict(zero_use)
        self._cumulative_request_metrics = dict(zero_req)
        self._cumulative_usage_metrics = dict(zero_use)
        self._ensure_auth()

    def _log_gateway_error(self, *, event: str, attempt: int, error: BaseException) -> None:
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
            source="anthropic",
            event=event,
            model_name=self.config.model_name,
            endpoint=self.config.anthropic_endpoint,
            attempt=attempt,
            error_type=type(error).__name__,
            error=str(error),
            status_code=getattr(response, "status_code", None),
            response_text=response_text,
        )

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        system_prompt, anth_messages = _serialize_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": anth_messages,
            "max_tokens": self.config.max_output_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    async def _query_async(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": self.config.anthropic_version,
        }

        last_error: ValueError | None = None
        raw_text = ""
        request_messages = list(messages)
        for attempt_index in range(MAX_JSON_PARSE_RETRIES + 1):
            payload = self._build_payload(request_messages)
            metrics_input = _metrics_input_from_anthropic(
                payload.get("system"), payload["messages"]
            )
            request_metrics = _request_metrics_from_serialized_input(metrics_input)
            self._last_request_metrics = dict(request_metrics)
            for key, value in request_metrics.items():
                self._cumulative_request_metrics[key] += value

            response_payload = None
            for rate_limit_attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
                        response = await client.post(
                            self.config.anthropic_endpoint,
                            headers=headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        response_payload = response.json()
                    break
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        self._log_gateway_error(
                            event="rate_limit_error", attempt=rate_limit_attempt + 1, error=exc
                        )
                        if rate_limit_attempt >= MAX_RATE_LIMIT_RETRIES:
                            raise
                        # Random wait in [MIN, MAX] seconds to desynchronize workers
                        # and let the org-level TPM bucket refill between retries.
                        delay = random.uniform(
                            RATE_LIMIT_BACKOFF_MIN_SECONDS,
                            RATE_LIMIT_BACKOFF_MAX_SECONDS,
                        )
                        # Retry-After can only extend the wait, never shorten it.
                        retry_after = _retry_after_seconds(exc)
                        if retry_after is not None and retry_after > delay:
                            delay = min(retry_after, RATE_LIMIT_BACKOFF_MAX_SECONDS * 2)
                        await asyncio.sleep(delay)
                        continue
                    if _is_transient_gateway_error(exc):
                        self._log_gateway_error(
                            event="transient_gateway_error",
                            attempt=rate_limit_attempt + 1,
                            error=exc,
                        )
                        if rate_limit_attempt >= MAX_TRANSIENT_GATEWAY_RETRIES:
                            raise
                        # Exponential backoff: 1.5, 3, 6, 12, 24, 30, 30, ...
                        await asyncio.sleep(
                            min(
                                TRANSIENT_BACKOFF_BASE_SECONDS * (2 ** rate_limit_attempt),
                                TRANSIENT_BACKOFF_CAP_SECONDS,
                            )
                        )
                        continue
                    self._log_gateway_error(
                        event="fatal_gateway_error", attempt=rate_limit_attempt + 1, error=exc
                    )
                    raise

            if response_payload is None:
                raise RuntimeError("Anthropic request returned no payload.")

            usage_metrics = _usage_from_anthropic_payload(response_payload)
            self._last_usage_metrics = dict(usage_metrics)
            for key, value in usage_metrics.items():
                self._cumulative_usage_metrics[key] += value

            raw_text = _extract_anthropic_text(response_payload)
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
                    request_messages.append(
                        self._format_repair_message(raw_text=raw_text, error=str(exc))
                    )
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
                    "anthropic_api_key": "<redacted>",
                },
                "usage": {**self._usage_snapshot()},
                "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            }
        }
