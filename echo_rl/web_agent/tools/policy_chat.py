"""Tiny helper that posts an OpenAI-compatible /chat/completions request to
the *current* policy via SkyRL's inference-engine HTTP endpoint.

The endpoint URL lives in ``$WEB_AGENT_POLICY_URL`` and the model name in
``$WEB_AGENT_POLICY_MODEL``. Both are exported by ``WebAgentEnvironment`` so
they are visible to ``python -c '...'`` snippets and to plain shell commands
the agent issues.

The tools in this package use this helper to call the *same* model that is
being trained, instead of routing through an external judge. That way the
model self-judges (image_qa, self_reflection) — no extra third-party token
spend, and the policy supervises itself.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

import httpx

_DEFAULT_TIMEOUT_S = 120
_DEFAULT_MAX_TOKENS = 1024


def policy_url() -> str:
    url = os.environ.get("WEB_AGENT_POLICY_URL", "").strip()
    if not url:
        raise RuntimeError(
            "WEB_AGENT_POLICY_URL is not set. The web-agent harness exports it "
            "to subprocesses when the policy's HTTP endpoint is enabled. If "
            "you're running this outside of training, set "
            "WEB_AGENT_POLICY_URL=http://host:port/chat/completions yourself."
        )
    # Be friendly about trailing path bits.
    if url.endswith("/v1/chat/completions") or url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1") or url.endswith("/v1/"):
        return url.rstrip("/") + "/chat/completions"
    return url.rstrip("/") + "/chat/completions"


def policy_model() -> str:
    return os.environ.get("WEB_AGENT_POLICY_MODEL", "").strip() or "policy"


def encode_image(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    mime, _ = mimetypes.guess_type(str(p))
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime or 'image/png'};base64,{encoded}"},
    }


def chat(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    model: str | None = None,
) -> str:
    """POST messages to the policy endpoint, return the assistant text."""
    payload = {
        "model": model or policy_model(),
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(policy_url(), json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"policy HTTP error {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ) from None
        body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"policy returned no choices: {json.dumps(body)[:500]}")
    message = choices[0].get("message", {}) or {}
    content = message.get("content")
    if isinstance(content, list):
        # Handle multi-part response content arrays.
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "\n".join(p for p in parts if p)
    return str(content or "").strip()


def parse_args_iter(argv: list[str], flag: str) -> list[str]:
    """Collect every value following each occurrence of ``flag`` in argv."""
    values: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            values.append(argv[i + 1])
            i += 2
            continue
        i += 1
    return values


def main(argv: list[str] | None = None) -> int:
    """``python -m echo_rl.web_agent.tools.policy_chat --message ...`` for ad-hoc use."""
    argv = list(argv or sys.argv[1:])
    messages_raw = parse_args_iter(argv, "--message")
    if not messages_raw:
        print("usage: --message '{\"role\": \"user\", \"content\": \"...\"}' [--message ...]", file=sys.stderr)
        return 2
    msgs = []
    for m in messages_raw:
        try:
            msgs.append(json.loads(m))
        except json.JSONDecodeError as exc:
            print(f"bad --message JSON: {exc}", file=sys.stderr)
            return 2
    print(chat(msgs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
