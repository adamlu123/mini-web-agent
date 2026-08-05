"""Generic TRAPI chat-completions backend for the GPT-5.x deployments.

Same wire protocol and Azure AD auth as :mod:`trapi_kimi_model`, but built for
the reasoning deployments served on TRAPI, e.g.::

    gpt-5.4_2026-03-05
    gpt-5.5_2026-04-24
    gpt-5.6-luna_2026-07-09
    gpt-5.6-sol_2026-07-09

Those deployments differ from Kimi in three ways, all handled here:

* ``max_tokens`` is rejected ("Unsupported parameter ... use
  ``max_completion_tokens`` instead"), so the token budget is emitted under the
  configured ``token_parameter``.
* They accept a top-level ``reasoning_effort`` knob.
* ``response_format.json_schema.strict`` must be ``false`` for the agent's action
  schema, which leaves ``bash_command``/``python_code`` optional and does not set
  ``additionalProperties: false``.

A ``deployment`` is what the harness calls an "endpoint": one TRAPI URL segment.
The batch runner pins exactly one deployment per task, so a single model instance
only ever talks to one of them.
"""

from __future__ import annotations

import random
import re
from typing import Any, Literal

from pydantic import field_validator

from miniswewebagent.models.trapi_kimi_model import (
    TrapiKimiModel,
    TrapiKimiModelConfig,
    _retry_after_seconds,
)

# Deployments available on the `gcr/shared` TRAPI instance for this project.
# Kept here so configs, the batch runner, and tests share one source of truth.
DEFAULT_TRAPI_DEPLOYMENTS = (
    "gpt-5.4_2026-03-05",
    "gpt-5.6-luna_2026-07-09",
    "gpt-5.5_2026-04-24",
    "gpt-5.6-sol_2026-07-09",
)

# TRAPI encodes the wait in the 429 body rather than a Retry-After header, e.g.
#   {"statusCode": 429, "message": "Token limit is exceeded. Try again in 6 seconds."}
#   {"status": 429, "error": "TRAPI: Rate Limit Exceeded, retry after 1 seconds. ..."}
_RETRY_HINT_PATTERN = re.compile(
    r"(?:try again in|retry after)\s+([0-9]+(?:\.[0-9]+)?)\s*second", re.IGNORECASE
)


def _retry_hint_from_body(exc: Exception) -> float | None:
    """Extract the wait TRAPI suggests in the 429 response body, in seconds."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        text = str(getattr(response, "text", "") or "")
    except Exception:
        return None
    match = _RETRY_HINT_PATTERN.search(text)
    if not match:
        return None
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return None


class TrapiModelConfig(TrapiKimiModelConfig):
    model_name: str = "gpt-5.4_2026-03-05"
    trapi_api_version: str = "2025-04-01-preview"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    # gpt-5.x rejects `max_tokens`; kept configurable for older deployments.
    token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    # Strict schema enforcement needs `additionalProperties: false` plus every
    # property in `required`, which the agent's action schema deliberately is not.
    strict_schema: bool = False
    max_output_tokens: int = 16000
    # These deployments are throttled by a per-deployment APIM token bucket that
    # refills in seconds, and the 429 body states exactly how long to wait. Honor
    # that hint instead of Kimi's blind 30-60s sleep, which wastes 5-10x per 429.
    rate_limit_min_backoff_seconds: float = 2.0
    rate_limit_max_backoff_seconds: float = 90.0
    rate_limit_jitter_seconds: float = 5.0

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _empty_effort_is_none(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value


class TrapiModel(TrapiKimiModel):
    """TRAPI backend for GPT-5.x deployments."""

    def __init__(self, *, config_class: type = TrapiModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _rate_limit_delay(self, exc: Exception, attempt: int) -> float:
        """Wait as long as TRAPI asks for, plus jitter, instead of a flat 30-60s.

        The bucket refills on the order of seconds, so a hinted wait gets the
        worker back in flight roughly an order of magnitude sooner. Jitter keeps
        workers that were throttled together from retrying in lockstep. Without a
        hint, fall back to bounded exponential backoff.
        """
        hint = _retry_after_seconds(exc)
        if hint is None:
            hint = _retry_hint_from_body(exc)
        if hint is not None:
            base = max(hint, self.config.rate_limit_min_backoff_seconds)
        else:
            base = self.config.rate_limit_min_backoff_seconds * (2**attempt)
        base = min(base, self.config.rate_limit_max_backoff_seconds)
        jitter = random.uniform(0.0, min(base, self.config.rate_limit_jitter_seconds))
        return min(base + jitter, self.config.rate_limit_max_backoff_seconds)

    def _response_format(self) -> dict[str, Any] | None:
        mode = (self.config.response_mode or "").lower()
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.config.schema_name,
                    "schema": self._response_schema(),
                    "strict": bool(self.config.strict_schema),
                },
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        from miniswewebagent.models.trapi_kimi_model import _serialize_chat_messages

        payload: dict[str, Any] = {
            "messages": _serialize_chat_messages(messages),
            self.config.token_parameter: self.config.max_output_tokens,
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        response_format = self._response_format()
        if response_format is not None:
            payload["response_format"] = response_format
        return payload
