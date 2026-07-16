import base64
import io
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)
import os
import backoff

def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_predication(response, mode):
    """Extract the prediction from the response."""
    if mode == "Autonomous_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "AgentTrek_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "WebVoyager_eval":
        if "FAILURE" in response:
            return 0
        else:
            return 1
    elif mode == "WebJudge_Online_Mind2Web_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "WebJudge_general_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _uses_max_completion_tokens(model_name: str) -> bool:
    return model_name.lower().startswith(("o1", "o3", "o4"))


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
        content_parts = (
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else [part for part in content if isinstance(part, dict)]
        )
        mapped_role = "developer" if role == "system" else role
        serialized.append(
            {
                "type": "message",
                "role": mapped_role,
                "content": [
                    _serialize_response_content_part(part, role=mapped_role)
                    for part in content_parts
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
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text", content.get("output_text"))
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


class OpenaiEngine():
    def __init__(
        self,
        api_key=None,
        stop=[],
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri = "",
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        self.endpoint_target_uri = endpoint_target_uri
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.client = None if endpoint_target_uri else OpenAI(api_key=api_key)

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError, httpx.HTTPError),
        max_tries=6,
        on_backoff=log_error
    )
    def generate(self, messages, max_new_tokens=2048, temperature=0, model=None, **kwargs):
        model = model if model else self.model
        if self.endpoint_target_uri:
            payload = {
                "model": model,
                "input": _serialize_response_input(messages),
                "max_output_tokens": max_new_tokens,
            }
            if not _uses_max_completion_tokens(model or ""):
                payload["temperature"] = temperature
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    self.endpoint_target_uri,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_keys[0]}",
                    },
                    json=payload,
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise httpx.HTTPStatusError(
                        f"{exc}; response_body={response.text[:4000]}",
                        request=exc.request,
                        response=exc.response,
                    ) from None
            return [_extract_response_text(response.json())]

        request = {"model": model, "messages": messages, **kwargs}
        if model and _uses_max_completion_tokens(model):
            request["max_completion_tokens"] = max_new_tokens
            request.setdefault("reasoning_effort", "low")
        else:
            request["max_tokens"] = max_new_tokens
            request["temperature"] = temperature
        assert self.client is not None
        response = self.client.chat.completions.create(**request)
        return [choice.message.content for choice in response.choices]