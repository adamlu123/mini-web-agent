"""TRAPI Kimi model backend.

Talks to Microsoft TRAPI's OpenAI-compatible `/chat/completions` endpoint
(e.g. `https://trapi.research.microsoft.com/{instance}/chat/completions`)
using an Azure AD bearer token obtained via `azure.identity`. Designed for
the `Kimi-K2.5_1` deployment on `gcr/shared`, but any TRAPI chat deployment
will work.

This backend mirrors `OpenRouterModel` (chat.completions shape) but adds
structured JSON output enforcement via `response_format` — defaulting to
strict `json_schema` so the model must return objects matching the agent's
action schema.
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, field_validator

from miniswewebagent.models.phyagi_model import (
    DEFAULT_JSON_FORMAT_ERROR_TEMPLATE,
    DEFAULT_OBSERVATION_TEMPLATE,
    MAX_JSON_PARSE_RETRIES,
    PhyagiModel,
    _is_rate_limit_error,
    _is_transient_gateway_error,
    _request_metrics_from_serialized_input,
    _safe_int,
    _validate_bash_command,
)
from miniswewebagent.utils.logging import append_runtime_log

# TRAPI-specific retry limits. The shared TRAPI pool can return 429s for
# extended windows under load, so we allow many more retries and longer
# backoffs than the phyagi defaults.
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
    headers = getattr(response, "headers", None)
    header = headers.get("retry-after") if headers else None
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except (TypeError, ValueError):
        return None


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
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": _safe_int(usage.get("prompt_tokens")),
        "output_tokens": _safe_int(usage.get("completion_tokens")),
        "total_tokens": _safe_int(usage.get("total_tokens")),
        "cached_input_tokens": _safe_int(prompt_details.get("cached_tokens")),
        "reasoning_output_tokens": _safe_int(completion_details.get("reasoning_tokens")),
    }


class TrapiKimiModelConfig(BaseModel):
    model_name: str = "Kimi-K2.5_1"
    trapi_instance: str = "gcr/shared"
    trapi_endpoint: str = "https://trapi.research.microsoft.com"
    trapi_api_version: str = "2024-10-21"
    trapi_scope: str = "api://trapi/.default"
    max_output_tokens: int = 4000
    request_timeout_seconds: int = 120
    error_log_path: Path | None = None
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    # "json_schema" (strict schema enforcement), "json_object" (free-form JSON),
    # or "none" (no enforcement; prompt-only).
    response_mode: str = "json_schema"
    format_error_template: str = DEFAULT_JSON_FORMAT_ERROR_TEMPLATE
    attach_observation_screenshot: bool = True
    schema_name: str = "playwright_step"

    @field_validator(
        "model_name",
        "trapi_instance",
        "trapi_endpoint",
        "trapi_api_version",
        "trapi_scope",
        "observation_template",
        "response_mode",
        "format_error_template",
        "schema_name",
        mode="before",
    )
    @classmethod
    def _normalize_optional_strings(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class TrapiKimiModel(PhyagiModel):
    """Model backend for TRAPI's chat-completions endpoint with JSON enforcement."""

    def __init__(self, *, config_class: type = TrapiKimiModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        zero_req = {k: 0 for k in ("message_count", "text_part_count", "image_part_count", "text_chars", "serialized_chars")}
        zero_use = {k: 0 for k in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_output_tokens")}
        self._last_request_metrics = dict(zero_req)
        self._last_usage_metrics = dict(zero_use)
        self._cumulative_request_metrics = dict(zero_req)
        self._cumulative_usage_metrics = dict(zero_use)
        self._token_provider = self._build_token_provider()

    def _build_token_provider(self):
        # Import lazily so this module can be imported in environments without azure.identity.
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )

        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        return get_bearer_token_provider(credential, self.config.trapi_scope)

    async def _get_bearer_token(self) -> str:
        # AzureCliCredential is sync; run it in a thread to avoid blocking the event loop.
        return await asyncio.get_running_loop().run_in_executor(None, self._token_provider)

    def _chat_completions_url(self) -> str:
        base = self.config.trapi_endpoint.rstrip("/")
        instance = self.config.trapi_instance.strip("/")
        deployment = self.config.model_name
        return (
            f"{base}/{instance}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.config.trapi_api_version}"
        )

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
            source="gateway",
            event=event,
            model_name=self.config.model_name,
            endpoint=self._chat_completions_url(),
            attempt=attempt,
            error_type=type(error).__name__,
            error=str(error),
            status_code=getattr(response, "status_code", None),
            response_text=response_text,
        )

    def _response_format(self) -> dict[str, Any] | None:
        mode = (self.config.response_mode or "").lower()
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.config.schema_name,
                    "schema": self._response_schema(),
                    "strict": True,
                },
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        # Azure OpenAI infers the deployment from the URL; no top-level "model" field.
        payload: dict[str, Any] = {
            "messages": _serialize_chat_messages(messages),
            "max_tokens": self.config.max_output_tokens,
        }
        response_format = self._response_format()
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _parse_model_output(self, raw_text: str) -> dict[str, Any]:
        # TRAPI Kimi always returns JSON under the enforcement modes; for
        # "none" we still fall back to JSON parsing since our agent contract
        # is JSON-shaped.
        from miniswewebagent.models.phyagi_model import (
            _repair_done_with_command,
            parse_json_output,
        )

        parsed = parse_json_output(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Unable to parse JSON object from TRAPI response.")
        return _repair_done_with_command(parsed)

    async def _query_async(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        last_error: ValueError | None = None
        raw_text = ""
        request_messages = list(messages)
        url = self._chat_completions_url()

        for attempt_index in range(MAX_JSON_PARSE_RETRIES + 1):
            payload = self._build_payload(request_messages)
            request_metrics = _request_metrics_from_serialized_input(payload["messages"])
            self._last_request_metrics = dict(request_metrics)
            for key, value in request_metrics.items():
                self._cumulative_request_metrics[key] += value

            response_payload = None
            for rate_limit_attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    token = await self._get_bearer_token()
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {token}",
                    }
                    async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
                        response = await client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        response_payload = response.json()
                    break
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        self._log_gateway_error(event="rate_limit_error", attempt=rate_limit_attempt + 1, error=exc)
                        if rate_limit_attempt >= MAX_RATE_LIMIT_RETRIES:
                            raise
                        # Random wait in [MIN, MAX] seconds to desynchronize workers
                        # and let the shared TRAPI TPM bucket refill between retries.
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
                        self._log_gateway_error(event="transient_gateway_error", attempt=rate_limit_attempt + 1, error=exc)
                        if rate_limit_attempt >= MAX_TRANSIENT_GATEWAY_RETRIES:
                            raise
                        # Exponential backoff: 1.5, 3, 6, 12, 24, 48, capped at 60.
                        await asyncio.sleep(
                            min(
                                TRANSIENT_BACKOFF_BASE_SECONDS * (2 ** rate_limit_attempt),
                                TRANSIENT_BACKOFF_CAP_SECONDS,
                            )
                        )
                        continue
                    self._log_gateway_error(event="fatal_gateway_error", attempt=rate_limit_attempt + 1, error=exc)
                    raise

            if response_payload is None:
                raise RuntimeError("TRAPI request returned no payload.")

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
                "config": {**self.config.model_dump(mode="json")},
                "usage": {**self._usage_snapshot()},
                "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            }
        }
