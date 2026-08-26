"""Azure OpenAI chat-completions backend (API-key auth, v1 surface).

Distinct from :mod:`azure_responses_model`, which targets the internal
``/responses`` deployments behind ``AzureCliCredential`` and a directory of
endpoint config files. This one speaks the plain OpenAI-compatible surface that
Azure exposes at ``{AZURE_OPENAI_ENDPOINT}/openai/v1/chat/completions`` and
authenticates with ``AZURE_OPENAI_API_KEY`` as a bearer token, so a laptop with
those two environment variables set can drive the same agent loop the phyagi
gateway drives in production.

Only two things differ from :class:`OpenAICompatibleModel`:

* auth/endpoint defaults come from ``AZURE_OPENAI_*`` instead of
  ``OPENAI_COMPATIBLE_*``;
* the payload sends ``max_completion_tokens``. Azure's gpt-5 family rejects
  ``max_tokens`` outright with HTTP 400
  ("Unsupported parameter: 'max_tokens' is not supported with this model"),
  and ``extra_body`` cannot help because it can add keys but not remove them.

Config (under ``model:`` in an agent yaml):

    model_class: azure_openai
    model_name: gpt-5.4          # the *deployment* name, not the catalog name
    response_mode: json_schema
    max_output_tokens: 16000
"""

from __future__ import annotations

import os
from typing import Any

from miniswewebagent.models.openai_compatible_model import (
    OpenAICompatibleModel,
    OpenAICompatibleModelConfig,
    _normalize_endpoint,
)


class AzureOpenAIModelConfig(OpenAICompatibleModelConfig):
    # The agent prompts in config/generation/* ask for a strict JSON object.
    response_mode: str = "json_schema"
    attach_observation_screenshot: bool = False
    max_output_tokens: int = 16000
    request_timeout_seconds: int = 240
    # Ask the server to constrain output to a JSON object. Azure supports this on
    # chat-completions; it is merged into the payload alongside the parser-side
    # json_schema response_mode, which stays responsible for the actual parse.
    force_json_object: bool = True


class AzureOpenAIModel(OpenAICompatibleModel):
    def __init__(self, *, config_class: type = AzureOpenAIModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _ensure_auth(self) -> None:
        endpoint = self.config.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not endpoint:
            raise ValueError(
                "azure_openai requires model.endpoint or AZURE_OPENAI_ENDPOINT "
                "(e.g. https://<resource>.openai.azure.com)."
            )
        endpoint = endpoint.rstrip("/")
        # Accept either the bare resource URL or one already pointing at the v1
        # surface; _normalize_endpoint appends /chat/completions to a /v1 tail.
        if "/openai/v1" not in endpoint and not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/openai/v1"
        self.config.endpoint = _normalize_endpoint(endpoint)
        self.config.openrouter_endpoint = self.config.endpoint

        api_key = self.config.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "azure_openai requires model.api_key or AZURE_OPENAI_API_KEY."
            )
        self.config.api_key = api_key
        self.config.openrouter_api_key = api_key

        if not self.config.model_name:
            self.config.model_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        if not self.config.model_name:
            raise ValueError(
                "azure_openai requires model.model_name (the Azure *deployment* name)."
            )

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = super()._build_payload(messages)
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        if self.config.force_json_object and "response_format" not in payload:
            payload["response_format"] = {"type": "json_object"}
        return payload
