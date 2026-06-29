from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from miniswewebagent.models.phyagi_model import _extract_response_text, text_part

# HTTP statuses that are commonly transient at the phyagi gateway and worth
# retrying. 401/403/404/413/422 are intentionally omitted — they will not
# succeed on retry.
_RETRYABLE_STATUS_CODES = frozenset({400, 408, 409, 425, 429, 500, 502, 503, 504})


def _build_prompt(question: str) -> str:
    return (
        "Answer the user's question using only visible evidence from the provided image or images. "
        "If the answer is not visible, say so instead of guessing.\n\n"
        f"Question: {question.strip()}"
    )


def _high_detail_image_part_from_path(image_path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type or 'image/png'};base64,{encoded}",
        "detail": "high",
    }


def _resolve_image_path(image_path: str, workspace_dir: str = "") -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        base_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return path


def _normalize_image_paths(
    *,
    image_path: Path | None = None,
    image_paths: list[Path] | tuple[Path, ...] | None = None,
) -> list[Path]:
    normalized = list(image_paths or [])
    if image_path is not None:
        normalized.insert(0, image_path)
    if not normalized:
        raise ValueError("At least one image path is required.")
    return normalized


def _gateway_config(args: argparse.Namespace) -> tuple[str, str, str]:
    endpoint = (
        args.endpoint
        or os.environ.get("WEB_AGENT_POLICY_URL", "")
        or os.environ.get("OPENAI_COMPATIBLE_ENDPOINT", "")
        or os.environ.get("OPENAI_GATEWAY_ENDPOINT", "")
        or "http://gateway.phyagi.net/api/responses"
    )
    endpoint = _normalize_endpoint(endpoint)
    model = (
        args.model
        or os.environ.get("WEB_AGENT_POLICY_MODEL", "")
        or os.environ.get("OPENAI_COMPATIBLE_MODEL", "")
        or os.environ.get("OPENAI_GATEWAY_MODEL", "gpt-5.4")
    )
    api_key = (
        args.api_key
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
        or os.environ.get("OPENAI_GATEWAY_API_KEY", "")
        or os.environ.get("PHYAGI_API_KEY", "")
    )
    if not api_key and _is_chat_completions_endpoint(endpoint):
        api_key = "dummy"
    if not api_key:
        raise RuntimeError("Missing OPENAI_GATEWAY_API_KEY or PHYAGI_API_KEY.")
    return api_key, endpoint, model


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
) -> httpx.Response:
    last_error: str = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(url, headers=headers, json=json_body)
        except httpx.TransportError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_attempts:
                raise
            delay = _sleep_backoff(attempt, base_delay)
            print(
                f"[image_qa] transport error on attempt {attempt}/{max_attempts}: "
                f"{last_error}; retrying in {delay:.2f}s",
                file=sys.stderr,
            )
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
            snippet = response.text[:500].replace("\n", " ") if response.text else ""
            delay = _sleep_backoff(attempt, base_delay)
            print(
                f"[image_qa] retryable HTTP {response.status_code} on attempt "
                f"{attempt}/{max_attempts}: {snippet}; retrying in {delay:.2f}s",
                file=sys.stderr,
            )
            continue

        response.raise_for_status()
        return response

    raise RuntimeError("image_qa retry loop exited without returning")  # unreachable


def run_image_qa(
    *,
    image_path: Path | None = None,
    image_paths: list[Path] | tuple[Path, ...] | None = None,
    question: str,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    max_attempts: int = 4,
    retry_base_delay: float = 1.0,
) -> dict[str, Any]:
    resolved_image_paths = _normalize_image_paths(image_path=image_path, image_paths=image_paths)
    if _is_chat_completions_endpoint(endpoint):
        return _run_image_qa_chat_completions(
            image_paths=resolved_image_paths,
            question=question,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            retry_base_delay=retry_base_delay,
        )

    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [text_part(_build_prompt(question))]
                + [_high_detail_image_part_from_path(path) for path in resolved_image_paths],
            }
        ],
        "max_output_tokens": 32000,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "image_qa_answer",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "answer": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "unknown": {"type": "boolean"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["answer", "evidence", "unknown", "confidence"],
                },
                "strict": True,
            }
        },
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
        )
        response_payload = response.json()

    raw_text = _extract_response_text(response_payload).strip()
    parsed = json.loads(raw_text)
    result = {
        "image_paths": [str(path) for path in resolved_image_paths],
        "question": question,
        **parsed,
    }
    if len(resolved_image_paths) == 1:
        result["image_path"] = str(resolved_image_paths[0])
    return result


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("image_qa response was not a JSON object")
    return parsed


def _run_image_qa_chat_completions(
    *,
    image_paths: list[Path],
    question: str,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    max_attempts: int,
    retry_base_delay: float,
) -> dict[str, Any]:
    system_prompt = (
        "Answer the user's question using only visible evidence from the provided image or images. "
        "If the answer is not visible, say so instead of guessing. Reply with one JSON object only, "
        "with keys answer, evidence, unknown, and confidence. Use evidence as a list of strings, "
        "unknown as a boolean, and confidence as a number from 0 to 1."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": _build_prompt(question)}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": _high_detail_image_part_from_path(path)["image_url"]},
        }
        for path in image_paths
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": 1024,
        "temperature": 0.0,
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
        )
        response_payload = response.json()

    parsed = _parse_json_object(_extract_chat_text(response_payload))
    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    elif not isinstance(evidence, list):
        evidence = [str(evidence)]
    result = {
        "image_paths": [str(path) for path in image_paths],
        "question": question,
        "answer": str(parsed.get("answer", "")),
        "evidence": evidence,
        "unknown": bool(parsed.get("unknown", False)),
        "confidence": parsed.get("confidence", 0),
    }
    if len(image_paths) == 1:
        result["image_path"] = str(image_paths[0])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a visual question about a local image and print JSON.")
    parser.add_argument(
        "--image",
        required=True,
        action="append",
        help="Path to an image file. Repeat --image to include multiple images.",
    )
    parser.add_argument("--question", required=True, help="Question to answer from the image.")
    parser.add_argument("--workspace-dir", default="", help="Optional base directory for relative image paths.")
    parser.add_argument("--model", default="", help="Override the gateway model name.")
    parser.add_argument("--endpoint", default="", help="Override the gateway endpoint.")
    parser.add_argument("--api-key", default="", help="Override the gateway API key.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Gateway request timeout.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Max total HTTP attempts before giving up (1 = no retry).",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        help="Base delay (seconds) for exponential backoff between retries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    image_paths = [_resolve_image_path(image_path, workspace_dir=args.workspace_dir) for image_path in args.image]
    api_key, endpoint, model = _gateway_config(args)
    result = run_image_qa(
        image_paths=image_paths,
        question=args.question,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        retry_base_delay=args.retry_base_delay,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
