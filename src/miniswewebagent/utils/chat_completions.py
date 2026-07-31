"""Payload helpers for OpenAI chat-completions judges (TRAPI, local vLLM).

The in-workspace judge tools build their prompts in the ``/responses`` content
shape (``input_text`` / ``input_image`` parts). Anything that speaks
chat-completions instead — Microsoft TRAPI, or the policy server itself under
``judge_model: policy`` — needs the same parts re-serialized and the reply text
extracted from ``choices[0].message``.
"""

from __future__ import annotations

from typing import Any


def serialize_chat_content_part(part: dict[str, Any]) -> dict[str, Any]:
    if part.get("type") == "input_image":
        image_url = {"url": part.get("image_url", "")}
        detail = part.get("detail")
        if detail:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}
    return {"type": "text", "text": part.get("text", "")}


def serialize_chat_user_content(user_content: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    serialized = [
        serialize_chat_content_part(part) for part in user_content if isinstance(part, dict)
    ]
    if serialized and all(part.get("type") == "text" for part in serialized):
        return "\n".join(part["text"] for part in serialized)
    return serialized


def extract_chat_text(payload: dict[str, Any]) -> str:
    """Return the assistant text, falling back to reasoning-only replies."""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text = "\n".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        if text:
            return text
    else:
        text = str(content or "").strip()
        if text:
            return text
    reasoning = message.get("reasoning_content", "")
    if isinstance(reasoning, list):
        return "\n".join(
            part.get("text", "") for part in reasoning if isinstance(part, dict)
        ).strip()
    return str(reasoning or "").strip()
