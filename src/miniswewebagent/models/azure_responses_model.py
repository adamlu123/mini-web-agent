"""Azure Responses-API model with round-robin across many deployments.

Subclasses :class:`PhyagiModel` and reuses every parsing / format-error /
JSON-schema enforcement bit. Only the network layer changes:

* Loads N endpoint config files (same JSON shape used by webeval_next /
  agento at ``endpoint_configs/<model>/prod/*.json``).
* Picks an endpoint at random (skipping currently blocklisted ones) for
  each request attempt.
* Authenticates with ``AzureCliCredential`` (token cached, refreshed on
  expiry) — no gateway, no shared bottleneck.
* On 5xx / 408 / 425 / 429 / 403 ("temporarily blocked") responses, the
  endpoint is added to a TTL blocklist and the next endpoint is tried.

Config (in agent yaml, under ``model:``):

    model_class: azure_responses
    endpoint_config_dir: /home/luyadong/sandbox/agento/endpoint_configs/gpt5.4/prod
    response_mode: json_schema
    max_output_tokens: 16000
    request_timeout_seconds: 120
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx
from azure.identity import AzureCliCredential
from pydantic import Field

from miniswewebagent.models.phyagi_model import (
    MAX_JSON_PARSE_RETRIES,
    MAX_RATE_LIMIT_RETRIES,
    MAX_TRANSIENT_GATEWAY_RETRIES,
    PhyagiModel,
    PhyagiModelConfig,
    _extract_response_text,
    _is_rate_limit_error,
    _is_transient_gateway_error,
    _request_metrics_from_serialized_input,
    _serialize_response_input,
    _usage_metrics_from_response_payload,
    _validate_bash_command,
)
from miniswewebagent.utils.logging import append_runtime_log


class AzureResponsesModelConfig(PhyagiModelConfig):
    endpoint_config_dir: Path = Path(
        "/home/luyadong/sandbox/agento/endpoint_configs/gpt5.4/prod"
    )
    endpoint_blocklist_seconds: float = 60.0
    azure_token_scope: str = "https://cognitiveservices.azure.com/.default"
    # Re-default to JSON schema; this model is built for it.
    response_mode: str = "json_schema"
    # No gateway key needed; relax the parent's required check.
    openai_gateway_api_key: str = "unused"


class _Endpoint:
    __slots__ = ("name", "azure_endpoint", "deployment", "api_version", "url",
                 "blocked_until")

    def __init__(self, cfg: dict[str, Any]):
        kwargs = cfg.get("CHAT_COMPLETION_KWARGS_JSON", {}) or {}
        self.azure_endpoint = str(kwargs["azure_endpoint"]).rstrip("/")
        self.deployment = str(kwargs["azure_deployment"])
        # The /openai/v1/responses surface only accepts api-version=preview
        # (other 2024/2025-MM-DD-preview versions return "API version not
        # supported"). The deployment is passed as ``model`` in the body.
        self.api_version = "preview"
        self.url = f"{self.azure_endpoint}/openai/v1/responses"
        self.name = self.deployment
        self.blocked_until: float = 0.0

    def is_available(self, now: float) -> bool:
        return self.blocked_until <= now


def _load_endpoint_configs(directory: Path) -> list[_Endpoint]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise RuntimeError(f"endpoint_config_dir does not exist: {directory}")
    endpoints: list[_Endpoint] = []
    for path in sorted(directory.glob("*.json")):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if cfg.get("CHAT_COMPLETION_PROVIDER", "").lower() != "azure":
                continue
            endpoints.append(_Endpoint(cfg))
        except Exception:
            continue
    if not endpoints:
        raise RuntimeError(f"No usable Azure endpoint configs in {directory}")
    return endpoints


class AzureResponsesModel(PhyagiModel):
    def __init__(self, *, config_class: type = AzureResponsesModelConfig, **kwargs):
        # Accept any phyagi-style kwargs unchanged (model_name etc.).
        super().__init__(config_class=config_class, **kwargs)
        self._endpoints: list[_Endpoint] = _load_endpoint_configs(
            self.config.endpoint_config_dir
        )
        self._credential = AzureCliCredential()
        self._token: str = ""
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    async def _bearer_token(self) -> str:
        # Refresh ~5 min before expiry. Token object is process-local; safe
        # to reuse across asyncio tasks.
        async with self._token_lock:
            now = time.time()
            if not self._token or now >= self._token_expires_at - 300:
                tok = await asyncio.to_thread(
                    self._credential.get_token, self.config.azure_token_scope
                )
                self._token = tok.token
                self._token_expires_at = float(tok.expires_on)
            return self._token

    def _pick_endpoint(self) -> _Endpoint:
        now = time.time()
        available = [ep for ep in self._endpoints if ep.is_available(now)]
        if not available:
            # All blocked — pick the one freeing up soonest, reset its block.
            soonest = min(self._endpoints, key=lambda e: e.blocked_until)
            soonest.blocked_until = 0.0
            available = [soonest]
        return random.choice(available)

    def _block_endpoint(self, ep: _Endpoint) -> None:
        ep.blocked_until = time.time() + self.config.endpoint_blocklist_seconds

    def _build_payload(self, messages: list[dict[str, Any]], deployment: str) -> dict[str, Any]:  # type: ignore[override]
        payload: dict[str, Any] = {
            "model": deployment,
            "input": _serialize_response_input(messages),
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.response_mode == "json_schema":
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "playwright_step",
                    "schema": self._response_schema(),
                    "strict": True,
                }
            }
        return payload

    async def _query_async(self, messages: list[dict[str, Any]]) -> dict[str, Any]:  # type: ignore[override]
        last_error: ValueError | None = None
        raw_text = ""
        request_messages = list(messages)
        parsed: dict[str, Any] | None = None

        for attempt_index in range(MAX_JSON_PARSE_RETRIES + 1):
            response_payload = None
            for net_attempt in range(MAX_TRANSIENT_GATEWAY_RETRIES + MAX_RATE_LIMIT_RETRIES + 1):
                ep = self._pick_endpoint()
                payload = self._build_payload(request_messages, ep.deployment)
                if net_attempt == 0:
                    request_metrics = _request_metrics_from_serialized_input(payload["input"])
                    self._last_request_metrics = dict(request_metrics)
                    for key, value in request_metrics.items():
                        self._cumulative_request_metrics[key] += value
                token = await self._bearer_token()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                params = {"api-version": ep.api_version}
                try:
                    async with httpx.AsyncClient(
                        timeout=self.config.request_timeout_seconds
                    ) as client:
                        response = await client.post(
                            ep.url, headers=headers, params=params, json=payload
                        )
                        response.raise_for_status()
                        response_payload = response.json()
                    break
                except Exception as exc:
                    transient = _is_transient_gateway_error(exc) or _is_rate_limit_error(exc)
                    self._log_gateway_error(
                        event="rate_limit_error" if _is_rate_limit_error(exc) else (
                            "transient_gateway_error" if transient else "fatal_gateway_error"
                        ),
                        attempt=net_attempt + 1,
                        error=exc,
                    )
                    if not transient:
                        raise
                    self._block_endpoint(ep)
                    await asyncio.sleep(min(0.5 * (net_attempt + 1), 5.0))
                    continue
            else:
                raise RuntimeError(
                    f"All endpoints exhausted after {net_attempt + 1} attempts"
                )

            if response_payload is None:
                raise RuntimeError("Azure request returned no payload.")

            usage_metrics = _usage_metrics_from_response_payload(response_payload)
            self._last_usage_metrics = dict(usage_metrics)
            for key, value in usage_metrics.items():
                self._cumulative_usage_metrics[key] += value

            raw_text = _extract_response_text(response_payload)
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

        assert parsed is not None
        actions: list[dict[str, Any]] = []
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

    def serialize(self) -> dict[str, Any]:  # type: ignore[override]
        base = super().serialize()
        base["model"]["endpoints"] = [ep.name for ep in self._endpoints]
        base["model"]["endpoint_config_dir"] = str(self.config.endpoint_config_dir)
        return base
