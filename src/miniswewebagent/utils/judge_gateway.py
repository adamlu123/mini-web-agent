"""Shared gateway resolution for the in-workspace judge tools.

``OPENAI_GATEWAY_ENDPOINT`` / ``OPENAI_GATEWAY_MODEL`` configure the *judge*
gateway used by ``self_reflection`` and ``image_qa``. They are **not** the policy
endpoint: the model under evaluation is configured by ``model.endpoint`` (and
mirrored into the workspace as ``OPENAI_COMPATIBLE_*``).

Both variables used to be read asymmetrically — the model came from the
environment while the endpoint was pinned to the ``/responses`` default. Pointing
the pair at a local vLLM server therefore asked the phyagi gateway for a
deployment that only exists locally, which 404s on every reflection attempt
instead of failing loudly. Resolution now honours both variables, and an endpoint
that clearly belongs to a policy server is rejected up front.

The one supported way to judge with the policy server is the explicit ``policy``
sentinel (``agent.judge_model: policy``, mirrored into the workspace as
``OPENAI_GATEWAY_MODEL=policy``): the tools then speak OpenAI chat-completions to
``OPENAI_COMPATIBLE_*`` / ``WEB_AGENT_POLICY_*`` — the same vLLM server that
generates the trajectory — instead of the ``/responses`` gateway.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

JUDGE_ENDPOINT_ENV = "OPENAI_GATEWAY_ENDPOINT"
JUDGE_MODEL_ENV = "OPENAI_GATEWAY_MODEL"

# Sentinel value for "judge with the policy server under evaluation".
POLICY_JUDGE_SENTINEL = "policy"
# Set by the harness when the run is policy-only. Any attempt to fall back to a
# /responses or TRAPI gateway is then a hard error rather than a silent reroute,
# so a misconfigured run fails on the first judge call instead of quietly
# scoring against a different model than the one under evaluation.
POLICY_ONLY_ENV = "MWA_JUDGE_POLICY_ONLY"
POLICY_ENDPOINT_ENVS = ("OPENAI_COMPATIBLE_ENDPOINT", "WEB_AGENT_POLICY_URL")
POLICY_MODEL_ENVS = ("OPENAI_COMPATIBLE_MODEL", "WEB_AGENT_POLICY_MODEL")
POLICY_API_KEY_ENVS = ("OPENAI_COMPATIBLE_API_KEY",)
# vLLM accepts any bearer token; keep parity with OpenAICompatibleModel.
DEFAULT_POLICY_API_KEY = "dummy"


@dataclass(frozen=True)
class PolicyJudgeTarget:
    """Chat-completions target for judging with the policy server."""

    endpoint: str
    model: str
    api_key: str


def resolve_judge_endpoint(endpoint: str = "") -> str:
    """Return the configured judge endpoint: explicit override, then environment.

    An empty result means "unset"; callers apply their own backend default.
    """
    return endpoint or os.environ.get(JUDGE_ENDPOINT_ENV, "")


def ensure_responses_endpoint(endpoint: str, *, model: str, tool: str) -> None:
    """Reject a chat-completions URL for a judge that speaks the ``/responses`` API."""
    if "/chat/completions" not in endpoint.lower():
        return
    raise RuntimeError(
        f"{tool}: judge endpoint {endpoint!r} is an OpenAI chat-completions URL, but this "
        f"judge speaks the /responses API (model={model!r}). {JUDGE_ENDPOINT_ENV} and "
        f"{JUDGE_MODEL_ENV} configure the judge gateway, not the policy server under "
        "evaluation. Point the policy at model.endpoint / OPENAI_COMPATIBLE_ENDPOINT and "
        "leave the judge on a /responses gateway, or set the judge model to "
        f"{POLICY_JUDGE_SENTINEL!r} to judge with the policy server on purpose."
    )


def is_policy_judge(model: str) -> bool:
    """True when ``model`` is the policy sentinel."""
    return (model or "").strip().lower() == POLICY_JUDGE_SENTINEL


def policy_judge_requested(model: str = "") -> bool:
    """True when the caller, or the environment, selected the policy sentinel.

    An explicit non-empty ``model`` always wins, so ``--model gpt-5.4`` still
    reaches the real gateway inside a policy-judge run.
    """
    if model:
        return is_policy_judge(model)
    return is_policy_judge(os.environ.get(JUDGE_MODEL_ENV, ""))


def ensure_policy_only_not_bypassed(*, model: str, tool: str) -> None:
    """Fail loudly when a policy-only run is about to use a non-policy backend.

    ``_gateway_config`` in both judge tools otherwise falls back to a
    ``/responses`` or TRAPI gateway using built-in defaults, which silently sends
    inference somewhere other than the vLLM server under evaluation.
    """
    if os.environ.get(POLICY_ONLY_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        return
    raise RuntimeError(
        f"{tool}: this run is policy-only ({POLICY_ONLY_ENV}=1, from agent.judge_model="
        f"{POLICY_JUDGE_SENTINEL!r}), but the judge resolved to a non-policy backend "
        f"(model={model or '<default>'!r}). Every call must go to the vLLM server. "
        f"Drop the explicit --model/--endpoint override, or unset {POLICY_ONLY_ENV} "
        "if routing this call elsewhere is intentional."
    )


def normalize_chat_completions_url(endpoint: str) -> str:
    """Accept ``host:port``, ``.../v1`` and ``.../v1/chat/completions`` alike."""
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/v1/chat/completions"


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def resolve_policy_judge(
    *,
    endpoint: str = "",
    model: str = "",
    api_key: str = "",
    tool: str,
) -> PolicyJudgeTarget:
    """Resolve the policy server as the judge target.

    ``endpoint`` / ``model`` / ``api_key`` are explicit overrides; the sentinel
    itself is ignored as a model name so ``--model policy`` still resolves the
    real served model name from the environment.
    """
    resolved_endpoint = endpoint.strip() or _first_env(POLICY_ENDPOINT_ENVS)
    if not resolved_endpoint:
        raise RuntimeError(
            f"{tool}: judge model is {POLICY_JUDGE_SENTINEL!r} but no policy endpoint is "
            f"configured. Set one of {', '.join(POLICY_ENDPOINT_ENVS)} (the harness mirrors "
            "model.endpoint into the workspace) or pass --endpoint."
        )
    if "/responses" in resolved_endpoint.lower():
        raise RuntimeError(
            f"{tool}: policy judge endpoint {resolved_endpoint!r} is a /responses gateway URL. "
            f"The {POLICY_JUDGE_SENTINEL!r} judge speaks OpenAI chat-completions against the "
            "policy server; point it at the vLLM endpoint instead."
        )

    resolved_model = "" if is_policy_judge(model) else model.strip()
    resolved_model = resolved_model or _first_env(POLICY_MODEL_ENVS)
    if not resolved_model:
        raise RuntimeError(
            f"{tool}: judge model is {POLICY_JUDGE_SENTINEL!r} but the served model name is "
            f"unknown. Set one of {', '.join(POLICY_MODEL_ENVS)} to the name the policy server "
            "advertises on /v1/models."
        )

    return PolicyJudgeTarget(
        endpoint=normalize_chat_completions_url(resolved_endpoint),
        model=resolved_model,
        api_key=api_key or _first_env(POLICY_API_KEY_ENVS) or DEFAULT_POLICY_API_KEY,
    )
