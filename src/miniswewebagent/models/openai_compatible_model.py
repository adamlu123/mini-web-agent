"""OpenAI-compatible chat-completions backend for local vLLM servers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import field_validator

from miniswewebagent.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig
from miniswewebagent.models.phyagi_model import (
    DEFAULT_OBSERVATION_TEMPLATE,
    DEFAULT_XML_FORMAT_ERROR_TEMPLATE,
)


class OpenAICompatibleModelConfig(OpenRouterModelConfig):
    model_name: str = ""
    api_key: str = ""
    endpoint: str = ""
    openrouter_api_key: str = ""
    openrouter_endpoint: str = ""
    openrouter_referer: str = ""
    openrouter_app_title: str = "mini-web-agent"
    max_output_tokens: int = 4000
    request_timeout_seconds: int = 120
    error_log_path: Path | None = None
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    response_mode: str = "sft_bash"
    format_error_template: str = DEFAULT_XML_FORMAT_ERROR_TEMPLATE
    attach_observation_screenshot: bool = False

    @field_validator(
        "model_name",
        "api_key",
        "endpoint",
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


class OpenAICompatibleModel(OpenRouterModel):
    def __init__(self, *, config_class: type = OpenAICompatibleModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        endpoint = self.config.endpoint or self.config.openrouter_endpoint or os.environ.get(
            "OPENAI_COMPATIBLE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"
        )
        if endpoint.endswith("/v1") or endpoint.endswith("/v1/"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        elif not endpoint.endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self.config.endpoint = endpoint
        self.config.openrouter_endpoint = endpoint
        if not self.config.model_name:
            self.config.model_name = os.environ.get("OPENAI_COMPATIBLE_MODEL", "policy")
        api_key = self.config.api_key or self.config.openrouter_api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "dummy")
        self.config.api_key = api_key
        self.config.openrouter_api_key = api_key

        zero_req = {k: 0 for k in ("message_count", "text_part_count", "image_part_count", "text_chars", "serialized_chars")}
        zero_use = {k: 0 for k in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_output_tokens")}
        self._last_request_metrics = dict(zero_req)
        self._last_usage_metrics = dict(zero_use)
        self._cumulative_request_metrics = dict(zero_req)
        self._cumulative_usage_metrics = dict(zero_use)

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

        from miniswewebagent.utils.logging import append_runtime_log

        append_runtime_log(
            self.config.error_log_path,
            source="openai_compatible",
            event=event,
            model_name=self.config.model_name,
            endpoint=self.config.endpoint,
            attempt=attempt,
            error_type=type(error).__name__,
            error=str(error),
            status_code=getattr(response, "status_code", None),
            response_text=response_text,
        )

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = super()._build_payload(messages)
        payload["model"] = self.config.model_name
        return payload

    async def _query_async(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return await super()._query_async(messages)

    def serialize(self) -> dict[str, Any]:
        return {
            "model": {
                "config": {
                    **self.config.model_dump(mode="json"),
                    "api_key": "<redacted>",
                    "openrouter_api_key": "<redacted>",
                },
                "usage": {**self._usage_snapshot()},
                "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            }
        }
