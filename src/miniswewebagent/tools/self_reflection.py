"""Self-reflection two-stage screenshot judge CLI.

Previously named ``two_stage_judge``; renamed to ``self_reflection``.

Mirrors the control flow of
``om2w_judge/methods/webjudge_online_mind2web_sandbox.py``'s
``_webjudge_online_mind2web_sandbox_eval`` but pushes every prompt out to the
caller (typically the LLM agent driving this harness).

Stage 1: for each screenshot, send a (system, user + image) pair to the
gateway and parse a 1-5 ``Score`` with a short ``Reasoning``.

Stage 2: drop every per-image ``Reasoning`` into the caller-provided final
user prompt template (via ``{image_reasonings}``), attach EVERY screenshot,
and make one final call that must end with ``Status: success`` or
``Status: failure``.

The CLI reads all of its config from a single JSON file so the agent can
prepare it in one turn and invoke the tool in the next. By default it uses
Microsoft TRAPI's Kimi-K2.5 deployment (``Kimi-K2.5_1`` on ``gcr/shared``)
via Azure AD. Explicit legacy overrides still route to the phyagi Responses
gateway.

Usage::

    python -m miniswewebagent.tools.self_reflection --config judge_config.json

JSON schema (paths relative to ``--workspace-dir`` or the CWD)::

    {
      "images": ["final_runs/run_001/screenshots/final_execution_1.png", ...],
      "image_judge_system_prompt":     "...",
      "image_judge_user_prompt":       "...",           // sent verbatim with each image
            "final_verdict_system_prompt":   "...",
            "final_verdict_user_prompt":     "...{action_history_log}...{image_reasonings}..."
    }

Any of the four prompt fields may instead be supplied via
``<field>_file`` variants pointing to a text file on disk (recommended when
prompts contain many literal braces or newlines).

The output JSON written to ``--output`` (or stdout) contains the per-image
records, the image path list, the final response, and
``predicted_label`` (``1`` for success, ``0`` for failure, ``null`` if the
``Status:`` line could not be parsed). Exit code: 0 if PASS, 1 otherwise.

By default, ``--num-evals`` parallel self-reflection evaluations are run.
The gate PASSES only when ALL N runs return ``predicted_label == 1``;
otherwise the written JSON contains one of the failed verdicts (preferring
an explicit ``predicted_label == 0``) plus ``num_evals``,
``all_predicted_labels``, ``chosen_eval_index``, and a compact
``all_eval_runs`` summary.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from miniswewebagent.models.phyagi_model import _extract_response_text, text_part
from miniswewebagent.utils.browser_evidence import (
    DEFAULT_BROWSER_STEPS_FILE,
    format_action_history,
    image_context,
    load_browser_steps,
    optional_file_digest,
    trajectory_evidence_digest,
    trajectory_images,
)
from miniswewebagent.utils.chat_completions import (
    extract_chat_text,
    serialize_chat_user_content,
)
from miniswewebagent.utils.judge_gateway import (
    ensure_policy_only_not_bypassed,
    POLICY_JUDGE_SENTINEL,
    ensure_responses_endpoint,
    policy_judge_requested,
    resolve_judge_endpoint,
    resolve_policy_judge,
)

DEFAULT_RESPONSES_MODEL = "gpt-5.4"
DEFAULT_RESPONSES_ENDPOINT = "http://gateway.phyagi.net/api/responses"
DEFAULT_TRAPI_MODEL = "Kimi-K2.5_1"
DEFAULT_TRAPI_BASE_ENDPOINT = "https://trapi.research.microsoft.com"
DEFAULT_TRAPI_INSTANCE = "gcr/shared"
DEFAULT_TRAPI_API_VERSION = "2024-10-21"
DEFAULT_TRAPI_SCOPE = "api://trapi/.default"
DEFAULT_MODEL = DEFAULT_TRAPI_MODEL
DEFAULT_ENDPOINT = DEFAULT_TRAPI_BASE_ENDPOINT
DEFAULT_IMAGE_PARSE_MAX_RETRIES = 3
DEFAULT_NUM_EVALS = 1

_RETRYABLE_STATUS_CODES = frozenset({400, 408, 409, 425, 429, 500, 502, 503, 504})

_PROMPT_FIELDS = (
    ("image_judge_system_prompt", True),
    ("image_judge_user_prompt", True),
    ("final_verdict_system_prompt", True),
    ("final_verdict_user_prompt", True),
)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_TRAPI_TOKEN_PROVIDERS: dict[str, Any] = {}


@dataclass(frozen=True)
class _GatewayConfig:
    backend: str
    endpoint: str
    model: str
    api_key: str


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resolve_image_path(image_path: str, workspace_dir: str = "") -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        base_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return path


def _final_execution_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"final_execution_(\d+)_", name)
    if match:
        return (int(match.group(1)), name)
    nums = re.findall(r"\d+", name)
    return (int(nums[0]) if nums else 0, name)


def _run_id_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"run_(\d+)", name)
    if match:
        return (int(match.group(1)), name)
    return (0, name)


def _sorted_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES],
        key=lambda path: _final_execution_sort_key(path.name),
    )


def _discover_latest_run_screenshots(
    final_runs_dir: Path,
) -> tuple[Path | None, list[Path]]:
    """Find the highest-numbered ``final_runs/run_<id>/screenshots`` dir and its images.

    Returns ``(run_dir_or_None, sorted_image_paths)``. Empty list if no images found.
    """
    if not final_runs_dir.exists() or not final_runs_dir.is_dir():
        return None, []
    candidates = sorted(
        (d for d in final_runs_dir.iterdir() if d.is_dir() and re.fullmatch(r"run_\d+", d.name)),
        key=lambda p: _run_id_sort_key(p.name),
    )
    # Walk from highest-numbered run downward and pick the first one with any screenshots.
    for run_dir in reversed(candidates):
        screenshots_dir = run_dir / "screenshots"
        images = _sorted_image_paths(screenshots_dir)
        if images:
            return run_dir, images
    return None, []


def _infer_run_dir_from_images(images: list[Path]) -> Path | None:
    run_dirs = {
        path.parent.parent.resolve()
        for path in images
        if path.parent.name == "screenshots"
    }
    if len(run_dirs) == 1:
        return next(iter(run_dirs))
    return None


def _resolve_artifact_dir(
    *,
    images: list[Path],
    discovered_run_dir: Path | None,
    output_path: str,
    workspace_dir: str,
) -> Path | None:
    candidates: list[Path] = []

    inferred_run_dir = _infer_run_dir_from_images(images)
    if inferred_run_dir is not None:
        candidates.append(inferred_run_dir)

    if discovered_run_dir is not None:
        candidates.append(discovered_run_dir.resolve())

    if output_path:
        candidates.append(Path(output_path).resolve().parent)

    base_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
    candidates.append(base_dir)

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if (candidate / "final_script_log.txt").is_file():
            return candidate

    return ordered_candidates[0] if ordered_candidates else None


def _load_action_history_log(artifact_dir: Path | None) -> str:
    if artifact_dir is None:
        return ""
    log_path = artifact_dir / "final_script_log.txt"
    if not log_path.is_file():
        return ""
    return log_path.read_text(encoding="utf-8").rstrip()


def _load_trajectory_scope(
    workspace_dir: Path,
    manifest: str | Path = DEFAULT_BROWSER_STEPS_FILE,
) -> dict[str, Any]:
    rows = load_browser_steps(workspace_dir, manifest)
    image_entries = trajectory_images(workspace_dir, rows)
    images = [path for path, _row in image_entries]
    contexts = {str(path): image_context(row) for path, row in image_entries}
    return {
        "rows": rows,
        "images": images,
        "image_contexts": contexts,
        "action_history_log": format_action_history(rows),
        "evidence_digest": trajectory_evidence_digest(workspace_dir, rows),
        "covered_through_browser_step": max(
            (int(row.get("browser_step") or 0) for row in rows), default=0
        ),
        "session_epochs": sorted(
            {int(row.get("session_epoch") or 0) for row in rows if row.get("session_epoch")}
        ),
    }


def _image_cache_key(
    image_path: Path,
    *,
    image_context_text: str,
    image_judge_system_prompt: str,
    image_judge_user_prompt: str,
    gateway_config: _GatewayConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    for value in (
        image_context_text,
        image_judge_system_prompt,
        image_judge_user_prompt,
        gateway_config.backend,
        gateway_config.endpoint,
        gateway_config.model,
    ):
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _load_image_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    if not isinstance(entries, dict):
        return {}
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def _write_image_cache(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"version": 1, "entries": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _render_final_verdict_user_prompt(
    template: str,
    *,
    image_reasonings: str,
    action_history_log: str,
) -> str:
    rendered = template
    if "{image_reasonings}" in template or "{action_history_log}" in template:
        try:
            rendered = template.format(
                image_reasonings=image_reasonings,
                action_history_log=action_history_log,
            )
        except KeyError as exc:
            raise ValueError(
                "Unknown placeholder in final_verdict_user_prompt: "
                f"{exc.args[0]!r}. Supported placeholders are "
                "{image_reasonings} and {action_history_log}; double any literal "
                "braces as {{ and }}."
            ) from exc

    additions: list[str] = []
    if "{action_history_log}" not in template and action_history_log:
        additions.append(f"Action history log:\n{action_history_log}")
    if "{image_reasonings}" not in template and image_reasonings:
        additions.append(f"Image reasonings:\n{image_reasonings}")
    if additions:
        rendered = f"{rendered.rstrip()}\n\n" + "\n\n".join(additions)
    return rendered


def _high_detail_image_part_from_path(image_path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type or 'image/png'};base64,{encoded}",
        "detail": "high",
    }


# ---------------------------------------------------------------------------
# Gateway HTTP helpers
# ---------------------------------------------------------------------------

def _looks_like_trapi_endpoint(endpoint: str) -> bool:
    normalized = endpoint.lower()
    return "trapi.research.microsoft.com" in normalized or "/openai/deployments/" in normalized


def _trapi_chat_completions_url(base_endpoint: str, *, model: str) -> str:
    if "/openai/deployments/" in base_endpoint:
        return base_endpoint
    base = base_endpoint.rstrip("/")
    instance = DEFAULT_TRAPI_INSTANCE.strip("/")
    return (
        f"{base}/{instance}/openai/deployments/{model}/chat/completions"
        f"?api-version={DEFAULT_TRAPI_API_VERSION}"
    )


def _use_legacy_responses_backend(*, endpoint: str, model: str) -> bool:
    if endpoint:
        return not _looks_like_trapi_endpoint(endpoint)
    if model and model != DEFAULT_TRAPI_MODEL:
        return True
    env_model = os.environ.get("OPENAI_GATEWAY_MODEL", "")
    if env_model and env_model != DEFAULT_TRAPI_MODEL:
        return True
    return False


def _gateway_config(*, api_key: str, endpoint: str, model: str) -> _GatewayConfig:
    if policy_judge_requested(model):
        # judge_model=policy: reflect with the very server that produced the
        # trajectory (vLLM), not the /responses judge gateway.
        target = resolve_policy_judge(
            endpoint=endpoint, model=model, api_key=api_key, tool="self_reflection"
        )
        return _GatewayConfig(
            backend="policy_chat",
            endpoint=target.endpoint,
            model=target.model,
            api_key=target.api_key,
        )

    endpoint = resolve_judge_endpoint(endpoint)
    ensure_policy_only_not_bypassed(model=model, tool="self_reflection")
    if _use_legacy_responses_backend(endpoint=endpoint, model=model):
        resolved_endpoint = endpoint or DEFAULT_RESPONSES_ENDPOINT
        resolved_model = model or os.environ.get("OPENAI_GATEWAY_MODEL", DEFAULT_RESPONSES_MODEL)
        ensure_responses_endpoint(
            resolved_endpoint, model=resolved_model, tool="self_reflection"
        )
        resolved_key = (
            api_key
            or os.environ.get("OPENAI_GATEWAY_API_KEY", "")
            or os.environ.get("PHYAGI_API_KEY", "")
        )
        if not resolved_key:
            raise RuntimeError(
                "Missing OPENAI_GATEWAY_API_KEY or PHYAGI_API_KEY for the legacy responses backend."
            )
        return _GatewayConfig(
            backend="responses",
            endpoint=resolved_endpoint,
            model=resolved_model,
            api_key=resolved_key,
        )

    resolved_model = model or DEFAULT_TRAPI_MODEL
    resolved_endpoint = _trapi_chat_completions_url(
        endpoint or DEFAULT_TRAPI_BASE_ENDPOINT,
        model=resolved_model,
    )
    return _GatewayConfig(
        backend="trapi_kimi",
        endpoint=resolved_endpoint,
        model=resolved_model,
        api_key=api_key or "",
    )


def _sleep_backoff(attempt: int, base_delay: float) -> float:
    delay = base_delay * (2 ** (attempt - 1))
    delay += random.uniform(0.0, delay * 0.25)
    time.sleep(delay)
    return delay


def _post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    max_attempts: int,
    base_delay: float,
    tag: str,
) -> httpx.Response:
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(url, headers=headers, json=json_body)
        except httpx.TransportError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_attempts:
                raise
            delay = _sleep_backoff(attempt, base_delay)
            print(
                f"[{tag}] transport error {attempt}/{max_attempts}: "
                f"{last_error}; retrying in {delay:.2f}s",
                file=sys.stderr,
            )
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
            snippet = response.text[:500].replace("\n", " ") if response.text else ""
            delay = _sleep_backoff(attempt, base_delay)
            print(
                f"[{tag}] retryable HTTP {response.status_code} {attempt}/{max_attempts}: "
                f"{snippet}; retrying in {delay:.2f}s",
                file=sys.stderr,
            )
            continue

        response.raise_for_status()
        return response

    raise RuntimeError("self_reflection retry loop exited without returning")


def _trapi_token_provider(scope: str):
    provider = _TRAPI_TOKEN_PROVIDERS.get(scope)
    if provider is not None:
        return provider

    try:
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TRAPI self_reflection requires azure-identity. Install it and run "
            "`az login --scope api://trapi/.default`, or pass a legacy responses "
            "model/endpoint override instead."
        ) from exc

    credential = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    )
    provider = get_bearer_token_provider(credential, scope)
    _TRAPI_TOKEN_PROVIDERS[scope] = provider
    return provider


def _resolve_trapi_bearer_token(api_key: str) -> str:
    if api_key:
        return api_key
    return _trapi_token_provider(DEFAULT_TRAPI_SCOPE)()


# ---------------------------------------------------------------------------
# Gateway call: plain message list -> text
# ---------------------------------------------------------------------------

def _call_responses_gateway(
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    gateway_config: _GatewayConfig,
    timeout_seconds: int,
    max_new_tokens: int,
    max_attempts: int,
    retry_base_delay: float,
    tag: str,
) -> str:
    payload: dict[str, Any] = {
        "model": gateway_config.model,
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [text_part(system_prompt)],
            },
            {
                "type": "message",
                "role": "user",
                "content": user_content,
            },
        ],
        "max_output_tokens": max_new_tokens,
    }

    with httpx.Client(timeout=timeout_seconds) as client:
        response = _post_with_retry(
            client,
            gateway_config.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {gateway_config.api_key}",
            },
            json_body=payload,
            max_attempts=max_attempts,
            base_delay=retry_base_delay,
            tag=tag,
        )
        response_payload = response.json()

    return _extract_response_text(response_payload).strip()


def _call_chat_completions(
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    gateway_config: _GatewayConfig,
    bearer_token: str,
    timeout_seconds: int,
    max_new_tokens: int,
    max_attempts: int,
    retry_base_delay: float,
    tag: str,
) -> str:
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": serialize_chat_user_content(user_content),
            },
        ],
        "max_tokens": max_new_tokens,
    }
    # TRAPI selects the deployment from the URL path and ignores a body model;
    # an OpenAI-compatible server (vLLM) requires it.
    if gateway_config.backend != "trapi_kimi":
        payload["model"] = gateway_config.model

    with httpx.Client(timeout=timeout_seconds) as client:
        response = _post_with_retry(
            client,
            gateway_config.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {bearer_token}",
            },
            json_body=payload,
            max_attempts=max_attempts,
            base_delay=retry_base_delay,
            tag=tag,
        )
        response_payload = response.json()

    return extract_chat_text(response_payload).strip()


def _call_gateway(
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    gateway_config: _GatewayConfig,
    timeout_seconds: int,
    max_new_tokens: int,
    max_attempts: int,
    retry_base_delay: float,
    tag: str,
) -> str:
    if gateway_config.backend in {"trapi_kimi", "policy_chat"}:
        bearer_token = (
            gateway_config.api_key
            if gateway_config.backend == "policy_chat"
            else _resolve_trapi_bearer_token(gateway_config.api_key)
        )
        return _call_chat_completions(
            system_prompt=system_prompt,
            user_content=user_content,
            gateway_config=gateway_config,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
            max_new_tokens=max_new_tokens,
            max_attempts=max_attempts,
            retry_base_delay=retry_base_delay,
            tag=tag,
        )
    return _call_responses_gateway(
        system_prompt=system_prompt,
        user_content=user_content,
        gateway_config=gateway_config,
        timeout_seconds=timeout_seconds,
        max_new_tokens=max_new_tokens,
        max_attempts=max_attempts,
        retry_base_delay=retry_base_delay,
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Parsing helpers (ported from webjudge_online_mind2web_sandbox.py)
# ---------------------------------------------------------------------------

def _parse_image_judge_response(response: str) -> tuple[str, int]:
    score_match = re.search(r"(?is)\bscore\b[^1-5]*([1-5])\b", response)
    reasoning_match = re.search(
        r"(?is)(?:\*\*?\s*reasoning\s*\*\*?|reasoning)\s*[:\-]\s*"
        r"(.*?)(?=\n\s*(?:\d+\.\s*)?(?:\*\*?\s*score\s*\*\*?|score)\s*[:\-]|\Z)",
        response,
    )

    if score_match and reasoning_match:
        reasoning = re.sub(r"\s+", " ", reasoning_match.group(1)).strip()
        return reasoning, int(score_match.group(1))

    try:
        payload = json.loads(response)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        score = payload.get("Score", payload.get("score"))
        reasoning = payload.get("Reasoning", payload.get("reasoning"))
        if (
            isinstance(score, int)
            and 1 <= score <= 5
            and isinstance(reasoning, str)
            and reasoning.strip()
        ):
            return re.sub(r"\s+", " ", reasoning).strip(), score

    raise ValueError("Could not parse image judge response")


def _parse_final_verdict(response: str) -> int | None:
    matches = list(re.finditer(r"(?i)status:\s*", response))
    if not matches:
        return None
    tail = response[matches[-1].end():].strip()
    m = re.match(r"""^[\'\"\u201c\u201d\u2018\u2019\s]*(success|failure)\b""", tail, re.IGNORECASE)
    if not m:
        return None
    return 1 if m.group(1).lower() == "success" else 0


# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------

async def _judge_one_image(
    *,
    image_path: Path,
    image_judge_system_prompt: str,
    image_judge_user_prompt: str,
    image_context_text: str,
    gateway_config: _GatewayConfig,
    timeout_seconds: int,
    max_attempts: int,
    retry_base_delay: float,
    max_new_tokens: int,
    max_parse_retries: int,
) -> dict[str, Any]:
    user_text = image_judge_user_prompt
    if image_context_text:
        user_text = f"{user_text.rstrip()}\n\n{image_context_text}"
    user_content = [
        text_part(user_text),
        _high_detail_image_part_from_path(image_path),
    ]

    last_response = ""
    last_error: BaseException | None = None
    for attempt in range(1, max_parse_retries + 1):
        last_response = await asyncio.to_thread(
            _call_gateway,
            system_prompt=image_judge_system_prompt,
            user_content=user_content,
            gateway_config=gateway_config,
            timeout_seconds=timeout_seconds,
            max_new_tokens=max_new_tokens,
            max_attempts=max_attempts,
            retry_base_delay=retry_base_delay,
            tag="self_reflection.image",
        )
        try:
            reasoning, score = _parse_image_judge_response(last_response)
            return {
                "image_path": str(image_path),
                "Response": last_response,
                "Score": score,
                "Reasoning": reasoning,
                "Attempts": attempt,
                "ParseFailed": False,
                "ImageContext": image_context_text,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"[self_reflection] parse attempt {attempt}/{max_parse_retries} failed for "
                f"{image_path}: {exc}",
                file=sys.stderr,
            )

    return {
        "image_path": str(image_path),
        "Response": last_response,
        "Score": 0,
        "Reasoning": "",
        "Attempts": max_parse_retries,
        "ParseFailed": True,
        "ParseError": str(last_error) if last_error is not None else "unknown",
        "ImageContext": image_context_text,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SelfReflectionResult:
    image_records: list[dict[str, Any]]
    image_paths: list[str]
    final_user_text: str
    final_system_msg: str
    final_response: str
    predicted_label: int | None  # 1 success, 0 failure, None unparsed
    model: str = ""
    endpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "predicted_label": self.predicted_label,
            "final_response": self.final_response,
            "final_user_text": self.final_user_text,
            "final_system_msg": self.final_system_msg,
            "image_paths": self.image_paths,
            "image_records": self.image_records,
        }


async def run_self_reflection_async(
    *,
    images: list[Path],
    image_judge_system_prompt: str,
    image_judge_user_prompt: str,
    final_verdict_system_prompt: str,
    final_verdict_user_prompt: str,
    action_history_log: str,
    image_contexts: dict[str, str] | None = None,
    cached_image_records: dict[str, dict[str, Any]] | None = None,
    image_cache_keys: dict[str, str] | None = None,
    max_image_parse_retries: int,
    final_max_new_tokens: int,
    image_max_new_tokens: int,
    gateway_config: _GatewayConfig,
    timeout_seconds: int,
    max_attempts: int,
    retry_base_delay: float,
) -> SelfReflectionResult:
    per_image = []
    image_contexts = image_contexts or {}
    cached_image_records = cached_image_records or {}
    image_cache_keys = image_cache_keys or {}
    if images:
        for path in images:
            cache_key = image_cache_keys.get(str(path), "")
            cached = cached_image_records.get(cache_key) if cache_key else None
            if cached is not None:
                record = dict(cached)
                record["image_path"] = str(path)
                record["CacheHit"] = True
            else:
                record = await _judge_one_image(
                    image_path=path,
                    image_judge_system_prompt=image_judge_system_prompt,
                    image_judge_user_prompt=image_judge_user_prompt,
                    image_context_text=image_contexts.get(str(path), ""),
                    gateway_config=gateway_config,
                    timeout_seconds=timeout_seconds,
                    max_attempts=max_attempts,
                    retry_base_delay=retry_base_delay,
                    max_new_tokens=image_max_new_tokens,
                    max_parse_retries=max_image_parse_retries,
                )
                record["CacheHit"] = False
            if cache_key:
                record["CacheKey"] = cache_key
            per_image.append(record)

    image_paths = [record["image_path"] for record in per_image]
    reasonings = [record["Reasoning"] or "" for record in per_image]

    reasonings_block = "\n".join(
        (
            f"{i + 1}. {image_contexts.get(image_paths[i], '').strip()}\n"
            f"   Image assessment: {text}"
        )
        for i, text in enumerate(reasonings)
    )

    final_user_text = _render_final_verdict_user_prompt(
        final_verdict_user_prompt,
        image_reasonings=reasonings_block,
        action_history_log=action_history_log,
    )

    user_content: list[dict[str, Any]] = [text_part(final_user_text)]
    if len(image_paths) <= 50:
        for path_str in image_paths:
            user_content.append(_high_detail_image_part_from_path(Path(path_str)))
    else:
        print(
            f"[self_reflection] skipping {len(image_paths)} final-stage images and using image reasonings only to stay within gateway limits",
            file=sys.stderr,
        )

    final_response = await asyncio.to_thread(
        _call_gateway,
        system_prompt=final_verdict_system_prompt,
        user_content=user_content,
        gateway_config=gateway_config,
        timeout_seconds=timeout_seconds,
        max_new_tokens=final_max_new_tokens,
        max_attempts=max_attempts,
        retry_base_delay=retry_base_delay,
        tag="self_reflection.final",
    )
    predicted_label = _parse_final_verdict(final_response)

    return SelfReflectionResult(
        image_records=list(per_image),
        image_paths=image_paths,
        final_user_text=final_user_text,
        final_system_msg=final_verdict_system_prompt,
        final_response=final_response,
        predicted_label=predicted_label,
        model=gateway_config.model,
        endpoint=gateway_config.endpoint,
    )


def run_self_reflection(**kwargs: Any) -> SelfReflectionResult:
    return asyncio.run(run_self_reflection_async(**kwargs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_prompt(cfg: dict[str, Any], key: str, *, required: bool) -> str | None:
    inline = cfg.get(key)
    file_key = f"{key}_file"
    file_path = cfg.get(file_key)
    if inline is not None and file_path is not None:
        raise ValueError(f"Provide only one of {key!r} or {file_key!r}, not both.")
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8")
    if inline is not None:
        return inline
    if required:
        raise ValueError(f"Missing required prompt: {key} (or {file_key}).")
    return None


def _load_config(config_arg: str) -> dict[str, Any]:
    if config_arg == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(config_arg).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage screenshot judge. Reads a JSON config describing images and "
            "prompts, calls TRAPI Kimi-K2.5 by default (with legacy responses "
            "gateway overrides still supported), and prints a JSON result with "
            "per-image records and the final verdict."
        )
    )
    parser.add_argument("--config", required=True, help="Path to JSON config, or '-' for stdin.")
    parser.add_argument("--workspace-dir", default="", help="Base directory for relative image paths.")
    parser.add_argument("--output", default="", help="Write JSON result to this path instead of stdout.")
    parser.add_argument(
        "--scope",
        choices=("latest-run", "trajectory"),
        default="latest-run",
        help="Judge the legacy latest final run or every saved incremental browser step.",
    )
    parser.add_argument(
        "--trajectory-manifest",
        default=DEFAULT_BROWSER_STEPS_FILE,
        help=f"Trajectory JSONL path (default: {DEFAULT_BROWSER_STEPS_FILE}).",
    )
    parser.add_argument(
        "--image-cache",
        default="",
        help="Cache per-image judgments. Trajectory scope defaults to reflection/image-cache.json.",
    )
    parser.add_argument(
        "--plan-file",
        default="plan.md",
        help="Plan path included in trajectory reflection freshness metadata.",
    )
    parser.add_argument(
        "--auto-latest-run",
        default="final_runs",
        help=(
            "When the config has no 'images' list, auto-discover screenshots from the "
            "highest-numbered `<workspace-dir>/<this-value>/run_<id>/screenshots` folder. "
            "Default: 'final_runs'. Pass '' (empty string) to disable auto-discovery."
        ),
    )
    parser.add_argument("--max-image-parse-retries", type=int, default=DEFAULT_IMAGE_PARSE_MAX_RETRIES)
    parser.add_argument(
        "--num-evals",
        type=int,
        default=DEFAULT_NUM_EVALS,
        help=(
            f"Number of parallel self-reflection evaluations to run. Default: {DEFAULT_NUM_EVALS}. "
            "All N must return predicted_label==1 for the gate to PASS; otherwise "
            "one of the failed verdicts is written to --output."
        ),
    )
    parser.add_argument("--image-max-new-tokens", type=int, default=2048)
    parser.add_argument("--final-max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Override the judge model or deployment. Defaults to TRAPI Kimi-K2.5 "
            f"({DEFAULT_TRAPI_MODEL}); explicit non-TRAPI overrides keep the legacy responses "
            f"backend. Pass {POLICY_JUDGE_SENTINEL!r} (or set "
            f"OPENAI_GATEWAY_MODEL={POLICY_JUDGE_SENTINEL}) to judge with the policy server "
            "itself over OpenAI chat-completions (OPENAI_COMPATIBLE_* / WEB_AGENT_POLICY_*)."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="",
        help=(
            "Override the judge endpoint. Defaults to the TRAPI base endpoint; "
            "explicit non-TRAPI endpoints keep the legacy responses backend."
        ),
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Override the bearer token or API key used by the selected backend.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=4, help="HTTP retry count per gateway call.")
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else Path.cwd().resolve()

    cfg = _load_config(args.config)

    prompts = {
        key: _resolve_prompt(cfg, key, required=required)
        for key, required in _PROMPT_FIELDS
    }

    trajectory_scope: dict[str, Any] | None = None
    image_contexts: dict[str, str] = {}
    if args.scope == "trajectory":
        trajectory_scope = _load_trajectory_scope(base_dir, args.trajectory_manifest)
        resolved_images = trajectory_scope["images"]
        image_contexts = trajectory_scope["image_contexts"]
        action_history_log = trajectory_scope["action_history_log"]
        discovered_run_dir = None
        print(
            f"[self_reflection] trajectory contains {len(trajectory_scope['rows'])} browser "
            f"steps and {len(resolved_images)} saved screenshots",
            file=sys.stderr,
        )
    else:
        images_config = cfg.get("images") or cfg.get("images_path") or []
        resolved_images = [
            _resolve_image_path(p, workspace_dir=args.workspace_dir) for p in images_config
        ]
        discovered_run_dir = _infer_run_dir_from_images(resolved_images)

        # If config did not provide images, fall back to the latest run's screenshots.
        if not resolved_images:
            discovered: list[Path] = []
            discovered_source = ""
            if args.auto_latest_run:
                auto_root = Path(args.auto_latest_run)
                if not auto_root.is_absolute():
                    auto_root = base_dir / auto_root
                auto_root = auto_root.resolve()
                discovered_run_dir, discovered = _discover_latest_run_screenshots(auto_root)
                if discovered_run_dir is not None:
                    discovered_source = str(discovered_run_dir / "screenshots")
            if discovered:
                resolved_images = discovered
                print(
                    f"[self_reflection] auto-discovered {len(resolved_images)} screenshots from "
                    f"{discovered_source}",
                    file=sys.stderr,
                )

        artifact_dir = _resolve_artifact_dir(
            images=resolved_images,
            discovered_run_dir=discovered_run_dir,
            output_path=args.output,
            workspace_dir=args.workspace_dir,
        )
        action_history_log = _load_action_history_log(artifact_dir)

    if not resolved_images:
        print(
            "[self_reflection] warning: no images provided; final stage will run without screenshot attachments.",
            file=sys.stderr,
        )

    if not action_history_log:
        print(
            "[self_reflection] warning: no action history found; final prompt will omit it.",
            file=sys.stderr,
        )

    gateway_config = _gateway_config(
        api_key=args.api_key, endpoint=args.endpoint, model=args.model
    )

    cache_path: Path | None = None
    if args.image_cache:
        cache_path = Path(args.image_cache)
        if not cache_path.is_absolute():
            cache_path = base_dir / cache_path
    elif args.scope == "trajectory":
        cache_path = base_dir / "reflection" / "image-cache.json"
    cached_image_records = _load_image_cache(cache_path)
    image_cache_keys = {
        str(path): _image_cache_key(
            path,
            image_context_text=image_contexts.get(str(path), ""),
            image_judge_system_prompt=prompts["image_judge_system_prompt"],
            image_judge_user_prompt=prompts["image_judge_user_prompt"],
            gateway_config=gateway_config,
        )
        for path in resolved_images
    }

    if args.num_evals < 1:
        print(
            f"ERROR: --num-evals must be >= 1 (got {args.num_evals}).",
            file=sys.stderr,
        )
        return 2

    print(
        f"[self_reflection] images={len(resolved_images)} backend={gateway_config.backend} "
        f"model={gateway_config.model} "
        f"num_evals={args.num_evals}",
        file=sys.stderr,
    )

    async def _run_all() -> list[SelfReflectionResult]:
        return await asyncio.gather(
            *(
                run_self_reflection_async(
                    images=resolved_images,
                    image_judge_system_prompt=prompts["image_judge_system_prompt"],
                    image_judge_user_prompt=prompts["image_judge_user_prompt"],
                    final_verdict_system_prompt=prompts["final_verdict_system_prompt"],
                    final_verdict_user_prompt=prompts["final_verdict_user_prompt"],
                    action_history_log=action_history_log,
                    image_contexts=image_contexts,
                    cached_image_records=cached_image_records,
                    image_cache_keys=image_cache_keys,
                    max_image_parse_retries=args.max_image_parse_retries,
                    final_max_new_tokens=args.final_max_new_tokens,
                    image_max_new_tokens=args.image_max_new_tokens,
                    gateway_config=gateway_config,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.max_attempts,
                    retry_base_delay=args.retry_base_delay,
                )
                for _ in range(args.num_evals)
            )
        )

    results = asyncio.run(_run_all())

    labels = [r.predicted_label for r in results]
    all_pass = all(lbl == 1 for lbl in labels)
    if all_pass:
        chosen_idx = 0
    else:
        # Prefer an explicit failure (label==0) over an unparsed verdict.
        chosen_idx = next(
            (i for i, lbl in enumerate(labels) if lbl == 0),
            next(
                (i for i, lbl in enumerate(labels) if lbl != 1),
                0,
            ),
        )

    chosen = results[chosen_idx]
    payload = chosen.to_dict()
    payload["num_evals"] = args.num_evals
    payload["all_predicted_labels"] = labels
    payload["chosen_eval_index"] = chosen_idx
    payload["all_eval_runs"] = [
        {
            "predicted_label": r.predicted_label,
            "final_response": r.final_response,
        }
        for r in results
    ]
    if trajectory_scope is not None:
        plan_path = Path(args.plan_file)
        if not plan_path.is_absolute():
            plan_path = base_dir / plan_path
        config_path = Path(args.config)
        if args.config != "-" and not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        payload.update(
            {
                "scope": "trajectory",
                "trajectory_manifest": str(args.trajectory_manifest),
                "covered_through_browser_step": trajectory_scope[
                    "covered_through_browser_step"
                ],
                "session_epochs": trajectory_scope["session_epochs"],
                "image_count": len(resolved_images),
                "evidence_digest": trajectory_scope["evidence_digest"],
                "plan_digest": optional_file_digest(plan_path.resolve()),
                "judge_config_digest": optional_file_digest(config_path.resolve())
                if args.config != "-"
                else "",
            }
        )

    if cache_path is not None:
        updated_cache = dict(cached_image_records)
        for record in chosen.image_records:
            cache_key = str(record.get("CacheKey") or "")
            if not cache_key or record.get("ParseFailed"):
                continue
            stored = dict(record)
            stored.pop("CacheHit", None)
            updated_cache[cache_key] = stored
        _write_image_cache(cache_path, updated_cache)

    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(f"Wrote result to {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(serialized)
        sys.stdout.write("\n")

    label = chosen.predicted_label
    if all_pass:
        print(
            f"JUDGE VERDICT: PASS (all {args.num_evals} evals predicted_label=1)",
            file=sys.stderr,
        )
        return 0
    if label == 0:
        print(
            f"JUDGE VERDICT: FAIL (labels={labels}; reporting failed eval #{chosen_idx})",
            file=sys.stderr,
        )
        return 1
    print(
        f"JUDGE VERDICT: UNPARSED (labels={labels}; reporting eval #{chosen_idx}; treating as FAIL)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
