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
from pydantic import BaseModel, Field, field_validator

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


# Fallback estimate for endpoints with no /tokenize: characters per token,
# measured over 40 OM2W request payloads against the Qwen3.5 tokenizer (observed
# range 2.36-4.09). Only used when an exact count is unavailable, and it is a
# heuristic, not a bound -- see _serialized_utf8_bytes for the safe bound.
MIN_CHARS_PER_TOKEN = 2.0

# Chat-template scaffolding (role markers, turn delimiters) added per message on
# top of the content tokens. Generous on purpose: it only pads the cheap
# short-circuit below, never the exact count.
TEMPLATE_TOKENS_PER_MESSAGE = 8


def _serialized_text_chars(serialized: list[dict[str, Any]]) -> int:
    total = 0
    for message in serialized:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"])
    return total


def _serialized_utf8_bytes(serialized: list[dict[str, Any]]) -> int:
    """UTF-8 byte length of the request text.

    This is a hard upper bound on the token count: the tokenizer is a byte-level
    BPE whose base vocabulary is the 256 single bytes, so no text ever encodes to
    more tokens than it has bytes, and merges only ever reduce the count.
    Character counts are NOT such a bound -- observed payloads reach 2.36
    chars/token, so a chars/N estimate can silently understate the real length.
    """
    total = 0
    for message in serialized:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content.encode("utf-8"))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"].encode("utf-8"))
    return total


def _tokenize_endpoints(chat_endpoint: str) -> list[str]:
    """Candidate /tokenize URLs for an OpenAI-compatible chat endpoint.

    vLLM serves /tokenize at the server root (the /v1 variant 404s), so try that
    first, then the versioned path for proxies that only expose /v1.
    """
    base = chat_endpoint.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    candidates = []
    if base.endswith("/v1"):
        candidates.append(f"{base[: -len('/v1')]}/tokenize")
    candidates.append(f"{base}/tokenize")
    return candidates


def _sft_state_assistant_content(message: dict[str, Any]) -> str | None:
    """Rebuild an assistant turn in the unified SFT state format.

    The harness stores only the parsed thought as message content; SFT
    checkpoints expect their own past turns verbatim, so replay the tags.
    """
    raw_response = (message.get("extra") or {}).get("raw_response")
    if not isinstance(raw_response, dict):
        return None
    thought = str(raw_response.get("thought") or "").strip()
    bash_command = str(raw_response.get("bash_command") or raw_response.get("python_code") or "").strip()
    done = "true" if bool(raw_response.get("done", False)) else "false"
    final_response = str(raw_response.get("final_response") or "").strip()
    return (
        f"<think>\n{thought}\n</think>\n"
        f"<bash>\n{bash_command}\n</bash>\n"
        f"<done>{done}</done>\n"
        f"<final_response>\n{final_response}\n</final_response>"
    )


def _serialize_chat_messages(
    messages: list[dict[str, Any]], *, response_mode: str = ""
) -> list[dict[str, Any]]:
    """Convert harness messages to OpenAI chat-completions format."""
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "exit":
            continue
        content = message.get("content", "")
        if role == "assistant" and response_mode == "sft_state":
            content = _sft_state_assistant_content(message) or content
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
    # Default to the neutral sampling temperature so a checkpoint's own
    # generation settings are not overridden; set explicitly (e.g. 0.0) for
    # greedy, reproducible decoding, or to null to omit the field entirely and
    # fall back to whatever the server defaults to.
    temperature: float | None = 1.0
    top_p: float | None = None
    seed: int | None = None
    # Verbatim extra chat-completions payload fields, for server-specific knobs
    # the config has no first-class setting for (vLLM `top_k`, `reasoning_effort`,
    # `chat_template_kwargs`, ...). Merged last, so it also overrides the fields
    # above when a key collides.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    error_log_path: Path | None = None
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    response_mode: str = "xml"
    format_error_template: str = DEFAULT_XML_FORMAT_ERROR_TEMPLATE
    attach_observation_screenshot: bool = True
    # See PhyagiModelConfig.sft_state_first_block.
    sft_state_first_block: bool = False

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
        # None until the first probe; then the working /tokenize URL, or "" when
        # the server has none and only the character estimate is available.
        self._tokenize_url: str | None = None
        self._ensure_auth()

    def prompt_tokens_within(self, messages: list[dict[str, Any]], budget: int) -> bool:
        """Whether the request built from `messages` fits in `budget` prompt tokens.

        Serializes exactly as `_build_payload` does, so the count reflects the
        `sft_state` assistant replay rather than the harness's stored content.
        A conservative character bound short-circuits the common case; only
        requests that might exceed the budget cost a `/tokenize` round trip.
        """
        if budget <= 0:
            return True
        serialized = _serialize_chat_messages(messages, response_mode=self.config.response_mode)
        # Skip the round trip only when the request cannot possibly exceed the
        # budget, using the byte bound rather than a character estimate.
        ceiling = _serialized_utf8_bytes(serialized) + TEMPLATE_TOKENS_PER_MESSAGE * len(serialized)
        if ceiling <= budget:
            return True
        exact = self._exact_prompt_tokens(serialized)
        if exact is None:
            # No exact counter available: fall back to the character heuristic.
            estimate = _serialized_text_chars(serialized) / MIN_CHARS_PER_TOKEN
            return estimate + TEMPLATE_TOKENS_PER_MESSAGE * len(serialized) <= budget
        return exact <= budget

    def _exact_prompt_tokens(self, serialized: list[dict[str, Any]]) -> int | None:
        """Prompt token count from the server's /tokenize endpoint, or None."""
        if self._tokenize_url == "":
            return None
        payload = {
            "model": self.config.model_name,
            "messages": serialized,
            "add_generation_prompt": True,
        }
        chat_template_kwargs = self.config.extra_body.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            payload["chat_template_kwargs"] = chat_template_kwargs
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
        }
        candidates = (
            [self._tokenize_url]
            if self._tokenize_url
            else _tokenize_endpoints(self.config.openrouter_endpoint)
        )
        for url in candidates:
            try:
                with httpx.Client(timeout=self.config.request_timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                count = _safe_int(response.json().get("count"))
            except Exception:
                continue
            if count > 0:
                self._tokenize_url = url
                return count
        self._tokenize_url = ""
        return None

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
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": _serialize_chat_messages(messages, response_mode=self.config.response_mode),
            "max_tokens": self.config.max_output_tokens,
        }
        for key in ("temperature", "top_p", "seed"):
            value = getattr(self.config, key, None)
            if value is not None:
                payload[key] = value
        payload.update(self.config.extra_body)
        return payload

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
