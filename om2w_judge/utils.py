import base64
import io
import os
import re
from typing import Any, Optional

import backoff
import httpx
from openai import APIConnectionError, APIError, OpenAI, RateLimitError


def _uses_max_completion_tokens(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    lowered = model_name.lower()
    return lowered.startswith("o")


def _serialize_response_content_part(part: dict[str, Any], *, role: str) -> dict[str, Any]:
    if part.get("type") == "image_url":
        image_url = part.get("image_url", "")
        if isinstance(image_url, dict):
            return {
                "type": "input_image",
                "image_url": str(image_url.get("url", "")),
                "detail": str(image_url.get("detail", "high") or "high"),
            }
        return {
            "type": "input_image",
            "image_url": str(image_url),
            "detail": str(part.get("detail", "high") or "high"),
        }

    text = str(part.get("text", ""))
    return {"type": "output_text" if role == "assistant" else "input_text", "text": text}


def _serialize_response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message["role"])
        content = message.get("content", "")
        if isinstance(content, str):
            serialized_content = [{"type": "text", "text": content}]
        else:
            serialized_content = [part for part in content if isinstance(part, dict)]

        mapped_role = "developer" if role == "system" else role
        serialized.append(
            {
                "type": "message",
                "role": mapped_role,
                "content": [
                    _serialize_response_content_part(part, role=mapped_role)
                    for part in serialized_content
                ],
            }
        )
    return serialized


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    texts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                texts.append(content["text"])
            elif isinstance(content.get("output_text"), str):
                texts.append(content["output_text"])
    return "\n".join(texts)


def encode_image(image):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_predication(response, mode):
    if mode in {
        "Autonomous_eval",
        "AgentTrek_eval",
        "WebJudge_Online_Mind2Web_eval",
        "WebJudge_Online_Mind2Web_Sandbox_eval",
        "WebJudge_Online_Mind2Web_Sandbox_eval_ctime",
        "WebJudge_Online_Mind2Web_Sandbox_WithThoughts_eval",
        "WebJudge_Online_Mind2Web_Sandbox_ThoughtsOnly_eval",
        "WebJudge_general_eval",
    }:
        try:
            matches = list(re.finditer(r"(?i)status:\s*", response))
            if matches:
                tail = response[matches[-1].end() :].strip()
                verdict_match = re.match(r'^[\'"“”‘’\s]*(success|failure)\b', tail, re.IGNORECASE)
                if verdict_match:
                    return 1 if verdict_match.group(1).lower() == "success" else 0
            return 0
        except Exception:
            return 0
    if mode == "WebVoyager_eval":
        return 0 if "FAILURE" in response else 1
    raise ValueError(f"Unknown mode: {mode}")


class OpenaiEngine:
    def __init__(
        self,
        api_key=None,
        stop=None,
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri="",
        **kwargs,
    ) -> None:
        del tokenizer, port, kwargs
        self.endpoint_target_uri = str(endpoint_target_uri or "")

        if not api_key:
            if self.endpoint_target_uri:
                api_key = (
                    os.getenv("OPENAI_GATEWAY_API_KEY")
                    or os.getenv("PHYAGI_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                )
            else:
                api_key = os.getenv("OPENAI_API_KEY", api_key)

        assert api_key, (
            "must pass on the api_key or set OPENAI_API_KEY in the environment"
            if not self.endpoint_target_uri
            else (
                "must pass on the api_key or set OPENAI_GATEWAY_API_KEY, "
                "PHYAGI_API_KEY, or OPENAI_API_KEY in the environment"
            )
        )
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop or []
        self.temperature = temperature
        self.model = model
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.api_key = self.api_keys[0]
        self.client = None if self.endpoint_target_uri else OpenAI(api_key=self.api_key)

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError, httpx.HTTPError),
        max_tries=3,
        on_backoff=log_error,
    )
    def generate(self, messages, max_new_tokens=1024, temperature=0, model=None, **kwargs):
        model = model if model else self.model
        if self.endpoint_target_uri:
            payload = {
                "model": model,
                "input": _serialize_response_input(messages),
                "max_output_tokens": max_new_tokens,
            }
            if not _uses_max_completion_tokens(model):
                payload["temperature"] = temperature
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    self.endpoint_target_uri,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                )
                response.raise_for_status()
                response_payload = response.json()
            return [_extract_response_text(response_payload)]

        create_kwargs = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        if _uses_max_completion_tokens(model):
            create_kwargs["max_completion_tokens"] = max_new_tokens
        else:
            create_kwargs["max_tokens"] = max_new_tokens
            create_kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**create_kwargs)
        return [choice.message.content for choice in response.choices]
