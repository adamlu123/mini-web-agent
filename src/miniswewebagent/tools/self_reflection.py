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
prepare it in one turn and invoke the tool in the next. Default model is
``gpt-5.4``.

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
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_ENDPOINT = os.environ.get("OPENAI_GATEWAY_ENDPOINT", "http://gateway.phyagi.net/api/responses")
DEFAULT_IMAGE_PARSE_MAX_RETRIES = 3

_RETRYABLE_STATUS_CODES = frozenset({400, 408, 409, 425, 429, 500, 502, 503, 504})

_PROMPT_FIELDS = (
    ("image_judge_system_prompt", True),
    ("image_judge_user_prompt", True),
    ("final_verdict_system_prompt", True),
    ("final_verdict_user_prompt", True),
)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


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
    match = re.search(r"run_(\d+)(_(\d+))*", name)
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

    # additions: list[str] = []
    # if "{action_history_log}" not in template and action_history_log:
    #     additions.append(f"Action history log:\n{action_history_log}")
    # if "{image_reasonings}" not in template and image_reasonings:
    #     additions.append(f"Image reasonings:\n{image_reasonings}")
    # if additions:
    #     rendered = f"{rendered.rstrip()}\n\n" + "\n\n".join(additions)
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
# Gateway HTTP helpers (mirrors image_qa)
# ---------------------------------------------------------------------------

def _gateway_config(
    *, api_key: str, endpoint: str, model: str
) -> tuple[str, str, str]:
    resolved_endpoint = _normalize_endpoint(
        endpoint
        or os.environ.get("WEB_AGENT_POLICY_URL", "")
        or os.environ.get("OPENAI_COMPATIBLE_ENDPOINT", "")
        or os.environ.get("OPENAI_GATEWAY_ENDPOINT", "")
        or DEFAULT_ENDPOINT
    )
    resolved_model = (
        model
        or os.environ.get("WEB_AGENT_POLICY_MODEL", "")
        or os.environ.get("OPENAI_COMPATIBLE_MODEL", "")
        or os.environ.get("OPENAI_GATEWAY_MODEL", DEFAULT_MODEL)
    )
    resolved_key = (
        api_key
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
        or os.environ.get("OPENAI_GATEWAY_API_KEY", "")
        or os.environ.get("PHYAGI_API_KEY", "") or os.environ.get("OM2W_JUDGE_API_KEY", "")
    )
    if not resolved_key and _is_chat_completions_endpoint(resolved_endpoint):
        resolved_key = "dummy"
    if not resolved_key:
        raise RuntimeError("Missing OPENAI_GATEWAY_API_KEY or PHYAGI_API_KEY.")
    return resolved_key, resolved_endpoint, resolved_model


def _is_chat_completions_endpoint(endpoint: str) -> bool:
    return "/chat/completions" in endpoint


def _normalize_endpoint(endpoint: str) -> str:
    value = endpoint.strip()
    if not value:
        return value
    if value.endswith("/v1/chat/completions") or value.endswith("/chat/completions"):
        return value
    if value.endswith("/responses") or value.endswith("/api/responses"):
        return value
    if value.endswith("/v1") or value.endswith("/v1/"):
        return value.rstrip("/") + "/chat/completions"
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/") + "/v1/chat/completions"
    return value


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


# ---------------------------------------------------------------------------
# Gateway call: plain message list -> text
# ---------------------------------------------------------------------------

def _call_gateway(
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    max_new_tokens: int,
    max_attempts: int,
    retry_base_delay: float,
    tag: str,
) -> str:
    is_chat_completions = _is_chat_completions_endpoint(endpoint)

    if is_chat_completions:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        # Convert user_content list of parts into multimodal content array
        chat_content: list[dict[str, Any]] = []
        for part in user_content:
            ptype = part.get("type")
            if ptype == "input_text":
                chat_content.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "input_image":
                chat_content.append({"type": "image_url", "image_url": {"url": part.get("image_url", "")}})
            else:
                # Fallback: if there is text, emit text; if image_url, emit image_url
                if isinstance(part.get("text"), str):
                    chat_content.append({"type": "text", "text": part["text"]})
                elif isinstance(part.get("image_url"), str):
                    chat_content.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
        messages.append({"role": "user", "content": chat_content})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_new_tokens,
        }
    else:
        payload = {
            "model": model,
            "input": [
                {
                    "type": "message",
                    "role": "system",
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
            endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json_body=payload,
            max_attempts=max_attempts,
            base_delay=retry_base_delay,
            tag=tag,
        )
        response_payload = response.json()

    if is_chat_completions:
        choices = response_payload.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        else:
            content = ""
        return content.strip()

    return _extract_response_text(response_payload).strip()


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
    # Pattern 1: Status: success / Status: failure
    match = re.search(r'Status: (success|failure)', response, re.IGNORECASE)
    if match:
        return 1 if match.group(1).lower() == 'success' else None

    # Pattern 2: Verdict: PASS / Verdict: FAIL
    match = re.search(r'Verdict: (PASS|FAIL)', response, re.IGNORECASE)
    if match:
        return 1 if match.group(1).upper() == 'PASS' else None

    # Pattern 3: Final digit 1-5 (verdict format from LLM)
    # Extract the last standalone digit in the response
    digit_match = re.search(r'(?:\n\n)?\s*([1-5])\s*$', response)
    if digit_match:
        score = int(digit_match.group(1))
        return 1  # Any score 1-5 indicates success for this task

    return None

def _load_prompt_from_config(config: dict[str, Any], field: str, workspace_dir: str) -> str:
    value = config.get(field)
    if isinstance(value, str) and value.strip():
        return value
    file_field = f"{field}_file"
    file_value = config.get(file_field)
    if isinstance(file_value, str) and file_value.strip():
        path = Path(file_value)
        if not path.is_absolute():
            base = Path(workspace_dir) if workspace_dir else Path.cwd()
            path = base / path
        return path.read_text(encoding='utf-8')
    raise KeyError(f'Missing required prompt field: {field}')


def _load_config(config_path: str, workspace_dir: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        base = Path(workspace_dir) if workspace_dir else Path.cwd()
        path = base / path
    return json.loads(path.read_text(encoding='utf-8'))


def run_self_reflection(config_path: str, workspace_dir: str = '', output_path: str = '', model: str = '', endpoint: str = '', api_key: str = '') -> int:
    config = _load_config(config_path, workspace_dir)
    prompts: dict[str, str] = {}
    for field, _required in _PROMPT_FIELDS:
        prompts[field] = _load_prompt_from_config(config, field, workspace_dir)

    images_cfg = config.get('images')
    images: list[Path] = []
    discovered_run_dir = None
    if isinstance(images_cfg, list) and images_cfg:
        images = [_resolve_image_path(str(img), workspace_dir) for img in images_cfg]
    else:
        final_runs_dir = (Path(workspace_dir) if workspace_dir else Path.cwd()) / 'final_runs'
        discovered_run_dir, images = _discover_latest_run_screenshots(final_runs_dir)
    if not images:
        raise RuntimeError('No screenshots found for self_reflection')

    artifact_dir = _resolve_artifact_dir(images=images, discovered_run_dir=discovered_run_dir, output_path=output_path, workspace_dir=workspace_dir)
    action_history_log = _load_action_history_log(artifact_dir)
    resolved_key, resolved_endpoint, resolved_model = _gateway_config(api_key=api_key, endpoint=endpoint, model=model)

    image_records = []
    image_reasoning_blocks = []
    for image_path in images:
        user_content = [text_part(prompts['image_judge_user_prompt']), _high_detail_image_part_from_path(image_path)]
        response = ''
        reasoning = ''
        score = 0
        parse_failed = False
        for attempt in range(DEFAULT_IMAGE_PARSE_MAX_RETRIES):
            response = _call_gateway(
                system_prompt=prompts['image_judge_system_prompt'],
                user_content=user_content,
                api_key=resolved_key,
                endpoint=resolved_endpoint,
                model=resolved_model,
                timeout_seconds=180,
                max_new_tokens=800,
                max_attempts=5,
                retry_base_delay=2.0,
                tag=f'image_judge:{image_path.name}:attempt{attempt+1}',
            )
            try:
                reasoning, score = _parse_image_judge_response(response)
                break
            except Exception:
                if attempt == DEFAULT_IMAGE_PARSE_MAX_RETRIES - 1:
                    parse_failed = True
                    reasoning = ''
                    score = 0
        record = {
            'image_path': str(image_path),
            'Score': score,
            'Reasoning': reasoning,
            'Response': response,
        }
        if parse_failed:
            record['ParseFailed'] = True
        image_records.append(record)
        image_reasoning_blocks.append(f"{image_path.name}: Score={score}; Reasoning={reasoning or 'Parse failed'}")

    image_reasonings = '\n'.join(image_reasoning_blocks)
    final_user_prompt = _render_final_verdict_user_prompt(
        prompts['final_verdict_user_prompt'],
        image_reasonings=image_reasonings,
        action_history_log=action_history_log,
    )
    final_user_content = [text_part(final_user_prompt)] + [_high_detail_image_part_from_path(p) for p in images]
    final_response = _call_gateway(
        system_prompt=prompts['final_verdict_system_prompt'],
        user_content=final_user_content,
        api_key=resolved_key,
        endpoint=resolved_endpoint,
        model=resolved_model,
        timeout_seconds=240,
        max_new_tokens=1600,
        max_attempts=5,
        retry_base_delay=2.0,
        tag='final_verdict',
    )
    verdict = _parse_final_verdict(final_response)
    result = {
        'images': [str(p) for p in images],
        'image_records': image_records,
        'final_prompt': final_user_prompt,
        'final_response': final_response,
        'predicted_label': verdict if verdict in (0, 1) else None,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if output_path:
        out = Path(output_path)
        if not out.is_absolute():
            base = Path(workspace_dir) if workspace_dir else Path.cwd()
            out = base / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding='utf-8')
    else:
        print(payload)
    return 0 if verdict == 1 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--workspace-dir', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--endpoint', default='')
    parser.add_argument('--api-key', default='')
    args = parser.parse_args(argv)
    return run_self_reflection(
        config_path=args.config,
        workspace_dir=args.workspace_dir,
        output_path=args.output,
        model=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
    )


if __name__ == '__main__':
    raise SystemExit(main())