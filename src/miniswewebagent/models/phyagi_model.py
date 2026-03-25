from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import httpx
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, field_validator

from miniswewebagent.exceptions import FormatError
from miniswewebagent.utils.runtime import run_async

MAX_JSON_PARSE_RETRIES = 3
MAX_RATE_LIMIT_RETRIES = 5
DEFAULT_OBSERVATION_TEMPLATE = """Observation:
Status: {{ 'ok' if observation.success else 'error' }}
URL: {{ observation.url }}
Title: {{ observation.title }}
{% if observation.exception %}Exception:
{{ observation.exception }}
{% endif %}{% if observation.console_output %}Console output:
{{ observation.console_output }}
{% endif %}{% if observation.aria_snapshot %}ARIA snapshot:
{{ observation.aria_snapshot }}
{% endif %}{% if observation.screenshot_path %}Screenshot path: {{ observation.screenshot_path }}
{% endif %}"""
DEFAULT_JSON_FORMAT_ERROR_TEMPLATE = """Format error:

{{ error }}

Please respond with strict JSON using exactly these fields:
- thought: short reasoning about the next step
- python_code: async Python Playwright code for exactly one step
- done: boolean indicating whether the task is complete
- final_response: final natural-language answer when done, otherwise empty
"""
DEFAULT_XML_FORMAT_ERROR_TEMPLATE = """Format error:

{{ error }}

Please respond using exactly these XML tags:
<response>
  <thought>...</thought>
  <python_code><![CDATA[
...python code...
]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
"""


def _is_rate_limit_error(exc: BaseException | None) -> bool:
    current: BaseException | None = exc
    while current is not None:
        status_code = getattr(current, "status_code", None)
        if status_code == 429:
            return True
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        text = str(current).lower()
        if "rate limit" in text or "ratelimit" in text or "too many requests" in text:
            return True
        current = current.__cause__ if isinstance(current.__cause__, BaseException) else None
    return False


def _extract_json_objects(raw: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start = None
    for index, char in enumerate(raw):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(raw[start : index + 1])
                start = None
    return objects


def parse_json_output(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for object_text in reversed(_extract_json_objects(raw)):
        try:
            return json.loads(object_text)
        except json.JSONDecodeError:
            continue
    raise ValueError("Unable to parse JSON output from gateway response.")


def _extract_xml_tag_values(raw: str, tag: str) -> list[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, raw, re.DOTALL | re.IGNORECASE)


def _strip_cdata(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("<![CDATA[") and stripped.endswith("]]>"):
        return stripped[len("<![CDATA[") : -len("]]>")]
    return stripped


def _parse_xml_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unable to parse boolean value from {value!r}.")


def parse_xml_output(raw: str) -> dict[str, Any]:
    thought_values = _extract_xml_tag_values(raw, "thought")
    python_code_values = _extract_xml_tag_values(raw, "python_code")
    done_values = _extract_xml_tag_values(raw, "done")
    final_response_values = _extract_xml_tag_values(raw, "final_response")
    if not (thought_values and python_code_values and done_values and final_response_values):
        raise ValueError("Unable to parse XML output from gateway response.")
    return {
        "thought": _strip_cdata(thought_values[-1]),
        "python_code": _strip_cdata(python_code_values[-1]),
        "done": _parse_xml_bool(_strip_cdata(done_values[-1])),
        "final_response": _strip_cdata(final_response_values[-1]),
    }


def text_part(text: str) -> dict[str, Any]:
    return {"type": "input_text", "text": text}


def image_part_from_path(path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(str(path))
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type or 'image/png'};base64,{encoded}",
        "detail": "high",
    }


def _serialize_gateway_content_part(part: dict[str, Any]) -> dict[str, Any]:
    if part.get("type") == "input_image":
        return {"type": "image_url", "image_url": {"url": part.get("image_url", "")}}
    return {"type": "text", "text": part.get("text", "")}


def _serialize_gateway_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "exit":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            serialized_content: str | list[dict[str, Any]] = content
        else:
            parts = [
                _serialize_gateway_content_part(part)
                for part in content
                if isinstance(part, dict)
            ]
            if parts and all(part.get("type") == "text" for part in parts):
                serialized_content = "\n".join(part.get("text", "") for part in parts)
            else:
                serialized_content = parts
        serialized.append({"role": role, "content": serialized_content})
    return serialized


def _extract_gateway_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
        return "\n".join(texts)
    return str(content)


class PhyagiModelConfig(BaseModel):
    model_name: str = "gpt-5.4"
    openai_gateway_api_key: str = ""
    openai_gateway_endpoint: str = "http://gateway.phyagi.net/api/chat/completions"
    openai_gateway_tier: str = "base"
    max_output_tokens: int = 4000
    request_timeout_seconds: int = 120
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    response_mode: str = "json_schema"
    format_error_template: str = DEFAULT_JSON_FORMAT_ERROR_TEMPLATE

    @field_validator(
        "model_name",
        "openai_gateway_api_key",
        "openai_gateway_endpoint",
        "openai_gateway_tier",
        "observation_template",
        "response_mode",
        "format_error_template",
        mode="before",
    )
    @classmethod
    def _normalize_optional_strings(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class PhyagiModel:
    def __init__(self, *, config_class: type = PhyagiModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if not self.config.openai_gateway_api_key:
            self.config.openai_gateway_api_key = os.environ.get(
                "OPENAI_GATEWAY_API_KEY",
                os.environ.get("PHYAGI_API_KEY", ""),
            )
        if not self.config.openai_gateway_api_key:
            raise RuntimeError("Missing OPENAI_GATEWAY_API_KEY or PHYAGI_API_KEY.")

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {"model_name": self.config.model_name, **kwargs}

    def format_message(self, **kwargs) -> dict[str, Any]:
        role = kwargs["role"]
        content = kwargs.get("content", "")
        extra = kwargs.get("extra", {})
        return {"role": role, "content": content, "extra": extra}

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        observation_messages: list[dict[str, Any]] = []
        for output in outputs:
            observation = output.get("observation", {})
            content = Template(self.config.observation_template, undefined=StrictUndefined).render(
                output=output,
                observation=observation,
                **(template_vars or {}),
            )

            parts: list[dict[str, Any]] = [text_part(content)]
            screenshot_path = observation.get("screenshot_path")
            if screenshot_path:
                parts.append(image_part_from_path(Path(screenshot_path)))

            observation_messages.append(
                self.format_message(role="user", content=parts, extra={"observation": observation})
            )
        return observation_messages

    def _response_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thought": {"type": "string"},
                "python_code": {"type": "string"},
                "done": {"type": "boolean"},
                "final_response": {"type": "string"},
            },
            "required": ["thought", "python_code", "done", "final_response"],
        }

    def _parse_model_output(self, raw_text: str) -> dict[str, Any]:
        if self.config.response_mode == "xml":
            return parse_xml_output(raw_text)
        return parse_json_output(raw_text)

    def _format_error(self, *, raw_text: str, error: str) -> FormatError:
        return FormatError(
            self.format_message(
                role="user",
                content=Template(self.config.format_error_template, undefined=StrictUndefined).render(
                    error=error,
                    model_response=raw_text,
                ),
                extra={
                    "interrupt_type": "FormatError",
                    "model_response": raw_text,
                },
            )
        )

    async def _query_async(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openai_gateway_api_key}",
        }
        payload = {
            "model": self.config.model_name,
            "messages": _serialize_gateway_messages(messages),
            "tier": self.config.openai_gateway_tier,
            "max_completion_tokens": self.config.max_output_tokens,
        }
        if self.config.response_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "playwright_step",
                    "strict": True,
                    "schema": self._response_schema(),
                },
            }

        last_error: ValueError | None = None
        raw_text = ""
        for _ in range(MAX_JSON_PARSE_RETRIES + 1):
            response_payload = None
            for rate_limit_attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
                        response = await client.post(
                            self.config.openai_gateway_endpoint,
                            headers=headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        response_payload = response.json()
                    break
                except Exception as exc:
                    if not _is_rate_limit_error(exc) or rate_limit_attempt >= MAX_RATE_LIMIT_RETRIES:
                        raise
                    await asyncio.sleep(min(5 * (rate_limit_attempt + 1), 30))

            if response_payload is None:
                raise RuntimeError("Gateway request returned no payload.")

            raw_text = _extract_gateway_text(response_payload)
            try:
                parsed = self._parse_model_output(raw_text)
                break
            except ValueError as exc:
                last_error = exc
        else:
            raise self._format_error(
                raw_text=raw_text,
                error=str(last_error or ValueError("Unable to parse model output.")),
            )

        actions = []
        python_code = parsed.get("python_code", "").strip()
        if python_code:
            actions.append({"python_code": python_code})

        return self.format_message(
            role="assistant",
            content=parsed.get("thought", ""),
            extra={
                "actions": actions,
                "done": bool(parsed.get("done", False)),
                "final_response": parsed.get("final_response", ""),
                "raw_response": parsed,
            },
        )

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        return run_async(self._query_async(messages))

    def serialize(self) -> dict[str, Any]:
        return {
            "model": {
                "config": {
                    **self.config.model_dump(mode="json"),
                    "openai_gateway_api_key": "<redacted>",
                },
                "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            }
        }
