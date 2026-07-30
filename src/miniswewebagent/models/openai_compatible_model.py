"""OpenAI-compatible chat-completions backend, used for local vLLM servers.

Reuses OpenRouterModel's payload/parse/retry machinery and only swaps the
auth story: a vLLM server needs no real API key, and the endpoint/model name
point at whatever checkpoint is being served (e.g. Qwen3.5-4B/9B).
"""

from __future__ import annotations

import os
from typing import Any

from miniswewebagent.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


class OpenAICompatibleModelConfig(OpenRouterModelConfig):
    model_name: str = ""
    endpoint: str = ""
    api_key: str = ""
    # SFT/RL checkpoints are served text-only and speak the unified state format.
    response_mode: str = "sft_state"
    attach_observation_screenshot: bool = False
    max_output_tokens: int = 4096
    request_timeout_seconds: int = 240


class OpenAICompatibleModel(OpenRouterModel):
    def __init__(self, *, config_class: type = OpenAICompatibleModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _ensure_auth(self) -> None:
        # A vLLM server accepts any bearer token; keep the OpenRouter fields in
        # sync so the inherited request path works unchanged.
        endpoint = self.config.endpoint or os.environ.get(
            "OPENAI_COMPATIBLE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"
        )
        self.config.endpoint = _normalize_endpoint(endpoint)
        self.config.openrouter_endpoint = self.config.endpoint
        if not self.config.model_name:
            self.config.model_name = os.environ.get("OPENAI_COMPATIBLE_MODEL", "policy")
        self.config.api_key = self.config.api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "dummy")
        self.config.openrouter_api_key = self.config.api_key

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
