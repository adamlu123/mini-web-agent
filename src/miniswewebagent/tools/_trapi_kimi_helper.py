"""Shared TRAPI Kimi backend for tool CLIs (image_qa, self_reflection).

When env var ``MWA_LLM_BACKEND=trapi_kimi`` is set, both tools route their
gateway calls through ``post_chat_completions`` below instead of the phyagi
responses API. Authentication uses ``AzureCliCredential`` /
``ManagedIdentityCredential`` against ``TRAPI_SCOPE``.

Env vars (defaults match the K2.5/K2.6 deployments on ``gcr/shared``):

- ``MWA_LLM_BACKEND``           : set to ``trapi_kimi`` to enable.
- ``TRAPI_KIMI_MODEL``          : deployment name (e.g. ``Kimi-K2.6_2026-04-20``).
- ``TRAPI_INSTANCE``            : default ``gcr/shared``.
- ``TRAPI_ENDPOINT``            : default ``https://trapi.research.microsoft.com``.
- ``TRAPI_API_VERSION``         : default ``2024-10-21``.
- ``TRAPI_SCOPE``               : default ``api://trapi/.default``.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

import httpx

_RETRYABLE_STATUS_CODES = frozenset({400, 408, 409, 425, 429, 500, 502, 503, 504})

_token_provider_lock = threading.Lock()
_token_provider = None


def should_use_trapi_kimi() -> bool:
    return os.environ.get("MWA_LLM_BACKEND", "").lower() == "trapi_kimi"


def _resolve_settings() -> dict[str, str]:
    return {
        "model": os.environ.get("TRAPI_KIMI_MODEL", "Kimi-K2.6_2026-04-20"),
        "instance": os.environ.get("TRAPI_INSTANCE", "gcr/shared"),
        "endpoint": os.environ.get("TRAPI_ENDPOINT", "https://trapi.research.microsoft.com").rstrip("/"),
        "api_version": os.environ.get("TRAPI_API_VERSION", "2024-10-21"),
        "scope": os.environ.get("TRAPI_SCOPE", "api://trapi/.default"),
    }


def _build_chat_completions_url(settings: dict[str, str]) -> str:
    return (
        f"{settings['endpoint']}/{settings['instance'].strip('/')}"
        f"/openai/deployments/{settings['model']}/chat/completions"
        f"?api-version={settings['api_version']}"
    )


def _get_token_provider(scope: str):
    global _token_provider
    with _token_provider_lock:
        if _token_provider is None:
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
            _token_provider = get_bearer_token_provider(credential, scope)
    return _token_provider


def _bearer_token(scope: str) -> str:
    return _get_token_provider(scope)()


def _convert_content_part(part: dict[str, Any]) -> dict[str, Any]:
    """Translate phyagi responses-API content parts to chat-completions shape."""
    ptype = part.get("type")
    if ptype == "input_image":
        return {"type": "image_url", "image_url": {"url": part.get("image_url", "")}}
    if ptype == "image_url":
        return part
    text = part.get("text", "")
    return {"type": "text", "text": text}


def _convert_messages(
    *, system_prompt: str, user_content: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    converted_parts = [_convert_content_part(p) for p in user_content if isinstance(p, dict)]
    if converted_parts and all(p.get("type") == "text" for p in converted_parts):
        messages.append({
            "role": "user",
            "content": "\n".join(p["text"] for p in converted_parts),
        })
    else:
        messages.append({"role": "user", "content": converted_parts})
    return messages


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _sleep_backoff(attempt: int, base_delay: float) -> float:
    delay = base_delay * (2 ** (attempt - 1))
    delay += random.uniform(0.0, delay * 0.25)
    delay = min(delay, 60.0)
    time.sleep(delay)
    return delay


def post_chat_completions(
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    max_new_tokens: int,
    timeout_seconds: int,
    max_attempts: int,
    retry_base_delay: float,
    tag: str = "trapi_kimi",
) -> str:
    """Synchronous TRAPI chat-completions call, returning the assistant text.

    Mirrors the retry semantics of the existing phyagi gateway helpers but
    against TRAPI's Azure-OpenAI-compatible endpoint.
    """
    import sys

    settings = _resolve_settings()
    url = _build_chat_completions_url(settings)
    messages = _convert_messages(system_prompt=system_prompt, user_content=user_content)
    body = {"messages": messages, "max_completion_tokens": max_new_tokens}

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            token = _bearer_token(settings["scope"])
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            delay = _sleep_backoff(attempt, retry_base_delay)
            print(
                f"[{tag}] auth error {attempt}/{max_attempts}: {type(exc).__name__}: {exc}; "
                f"retrying in {delay:.2f}s",
                file=sys.stderr,
            )
            continue

        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            delay = _sleep_backoff(attempt, retry_base_delay)
            print(
                f"[{tag}] transport error {attempt}/{max_attempts}: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:.2f}s",
                file=sys.stderr,
            )
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
            snippet = response.text[:500].replace("\n", " ") if response.text else ""
            delay = _sleep_backoff(attempt, retry_base_delay)
            # 429 from TRAPI can have long backoffs; honor Retry-After if longer.
            retry_after = response.headers.get("retry-after")
            try:
                ra = float(retry_after) if retry_after else 0.0
            except (TypeError, ValueError):
                ra = 0.0
            if ra > delay:
                time.sleep(min(ra, 60.0))
            print(
                f"[{tag}] retryable HTTP {response.status_code} {attempt}/{max_attempts}: "
                f"{snippet}; retried after backoff",
                file=sys.stderr,
            )
            continue

        response.raise_for_status()
        return _extract_chat_text(response.json()).strip()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{tag}: retry loop exited without returning")
