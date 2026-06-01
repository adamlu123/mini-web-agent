from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
import openai
from jinja2 import StrictUndefined, Template
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator

from miniswewebagent.exceptions import FormatError
from miniswewebagent.utils.logging import append_runtime_log
from miniswewebagent.utils.runtime import run_async

MAX_JSON_PARSE_RETRIES = 3
MAX_RATE_LIMIT_RETRIES = 5
MAX_TRANSIENT_GATEWAY_RETRIES = 5
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
- bash_command: exactly one shell command for local-workspace tasks
- python_code: legacy field for browser tasks; still accepted for backward compatibility
- done: boolean indicating whether the task is complete
- final_response: final natural-language answer when done, otherwise empty
"""
DEFAULT_XML_FORMAT_ERROR_TEMPLATE = """Format error:

{{ error }}

Please respond using exactly these XML tags:
<response>
  <thought>...</thought>
  <bash_command><![CDATA[
...exactly one shell command...
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>

Return XML only. Do not include markdown fences or any prose before or after the `<response>` block.
`<python_code>` is still accepted for backward compatibility, but prefer `<bash_command>` for local-workspace tasks.
"""


def _is_rate_limit_error(exc: BaseException | None) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, openai.RateLimitError):
            return True
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


def _is_transient_gateway_error(exc: BaseException | None) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
            return True
        if isinstance(current, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)):
            return True
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int) and status_code in {408, 409, 425, 500, 502, 503, 504}:
            return True
        response = getattr(current, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int) and response_status in {408, 409, 425, 500, 502, 503, 504}:
            return True
        text = str(current).lower()
        if any(
            needle in text
            for needle in (
                "bad gateway",
                "gateway timeout",
                "server disconnected",
                "temporary failure",
                "temporarily unavailable",
                "connection reset",
                "connection aborted",
                "timed out",
                # Transient phyagi-gateway upstream flake that surfaces as a 404
                # NotFoundError ("upstream.failed - The upstream model returned
                # an error. Try again in a moment." / "Azure OpenAI Responses
                # API is not enabled in this region."). These are intermittent
                # and clear on retry, so treat them as transient. Kept specific
                # to avoid retrying genuine not-found errors (e.g. bad model id).
                "upstream.failed",
                "try again in a moment",
                "responses api is not enabled",
            )
        ):
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
        return stripped[len("<![CDATA[") : -len("]]>")].strip()
    return stripped


def _parse_xml_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unable to parse boolean value from {value!r}.")


def _repair_done_with_command(parsed: dict[str, Any]) -> dict[str, Any]:
    bash_command = str(parsed.get("bash_command", "") or "").strip()
    python_code = str(parsed.get("python_code", "") or "").strip()
    if (bash_command or python_code) and bool(parsed.get("done", False)):
        repaired = dict(parsed)
        repaired["done"] = False
        return repaired
    return parsed


def _validate_bash_command(command: str) -> None:
    result = subprocess.run(
        ["/bin/bash", "-n"],
        input=command,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode == 0:
        return
    error = (result.stderr or result.stdout or "bash syntax check failed").strip()
    raise ValueError(f"Invalid bash_command syntax: {error}")


def parse_xml_output(raw: str) -> dict[str, Any]:
    thought_values = _extract_xml_tag_values(raw, "thought")
    bash_command_values = _extract_xml_tag_values(raw, "bash_command")
    python_code_values = _extract_xml_tag_values(raw, "python_code")
    done_values = _extract_xml_tag_values(raw, "done")
    final_response_values = _extract_xml_tag_values(raw, "final_response")
    if not (thought_values and done_values and final_response_values):
        raise ValueError("Unable to parse XML output from gateway response.")
    done = _parse_xml_bool(_strip_cdata(done_values[-1]))
    if not (bash_command_values or python_code_values or done):
        raise ValueError("Unable to parse XML output from gateway response.")
    return {
        "thought": _strip_cdata(thought_values[-1]),
        "bash_command": _strip_cdata(bash_command_values[-1]) if bash_command_values else "",
        "python_code": _strip_cdata(python_code_values[-1]) if python_code_values else "",
        "done": done,
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


def _serialize_response_content_part(part: dict[str, Any], *, role: str) -> dict[str, Any]:
    if part.get("type") == "input_image":
        return {
            "type": "input_image",
            "image_url": part.get("image_url", ""),
            "detail": part.get("detail", "high"),
        }
    text = part.get("text", "")
    if role == "assistant":
        return {"type": "output_text", "text": text}
    return {"type": "input_text", "text": text}


def _serialize_response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "exit":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            serialized_content = [text_part(content)]
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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _request_metrics_from_serialized_input(serialized_input: list[dict[str, Any]]) -> dict[str, int]:
    message_count = len(serialized_input)
    text_part_count = 0
    image_part_count = 0
    text_chars = 0

    for item in serialized_input:
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            part_type = content.get("type")
            if part_type in {"input_text", "output_text"}:
                text_part_count += 1
                text_chars += len(str(content.get("text", "") or ""))
            elif part_type == "input_image":
                image_part_count += 1

    serialized_chars = len(json.dumps(serialized_input, ensure_ascii=False))

    return {
        "message_count": message_count,
        "text_part_count": text_part_count,
        "image_part_count": image_part_count,
        "text_chars": text_chars,
        "serialized_chars": serialized_chars,
    }


def _usage_metrics_from_response_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}

    return {
        "input_tokens": _safe_int(usage.get("input_tokens")),
        "output_tokens": _safe_int(usage.get("output_tokens")),
        "total_tokens": _safe_int(usage.get("total_tokens")),
        "cached_input_tokens": _safe_int(input_details.get("cached_tokens")),
        "reasoning_output_tokens": _safe_int(output_details.get("reasoning_tokens")),
    }


class PhyagiModelConfig(BaseModel):
    model_name: str = "gpt-5.4"
    openai_gateway_api_key: str = ""
    openai_gateway_endpoint: str = "https://gateway.phyagi.net/api"
    openai_gateway_tier: str = "base"
    cache_ttl: int | None = None
    session_id: str = ""
    strict_session: bool = False
    max_output_tokens: int = 4000
    request_timeout_seconds: int = 120
    error_log_path: Path | None = None
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    response_mode: str = "xml"
    format_error_template: str = DEFAULT_XML_FORMAT_ERROR_TEMPLATE
    attach_observation_screenshot: bool = True

    @field_validator(
        "model_name",
        "openai_gateway_api_key",
        "openai_gateway_endpoint",
        "openai_gateway_tier",
        "session_id",
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
        self._last_request_metrics: dict[str, int] = {
            "message_count": 0,
            "text_part_count": 0,
            "image_part_count": 0,
            "text_chars": 0,
            "serialized_chars": 0,
        }
        self._last_usage_metrics: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        self._cumulative_request_metrics: dict[str, int] = {
            "message_count": 0,
            "text_part_count": 0,
            "image_part_count": 0,
            "text_chars": 0,
            "serialized_chars": 0,
        }
        self._cumulative_usage_metrics: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        default_tier = PhyagiModelConfig.model_fields["openai_gateway_tier"].default
        env_tier = os.environ.get(
            "OPENAI_GATEWAY_TIER",
            os.environ.get("PHYAGI_TIER", ""),
        )
        if env_tier and self.config.openai_gateway_tier == default_tier:
            self.config.openai_gateway_tier = env_tier

        if not self.config.openai_gateway_api_key:
            self.config.openai_gateway_api_key = os.environ.get(
                "OPENAI_GATEWAY_API_KEY",
                os.environ.get("PHYAGI_API_KEY", ""),
            )
        if not self.config.openai_gateway_api_key:
            raise RuntimeError("Missing OPENAI_GATEWAY_API_KEY or PHYAGI_API_KEY.")

        if self.config.cache_ttl is None:
            env_ttl = os.environ.get("PHYAGI_CACHE_TTL", "").strip()
            if env_ttl:
                try:
                    self.config.cache_ttl = int(env_ttl)
                except ValueError:
                    pass
        if not self.config.session_id:
            self.config.session_id = os.environ.get("PHYAGI_SESSION_ID", "")
        if not self.config.strict_session:
            env_strict = os.environ.get("PHYAGI_STRICT_SESSION", "").strip().lower()
            if env_strict in {"1", "true", "yes"}:
                self.config.strict_session = True

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "last_request_message_count": self._last_request_metrics["message_count"],
            "last_request_text_part_count": self._last_request_metrics["text_part_count"],
            "last_request_image_part_count": self._last_request_metrics["image_part_count"],
            "last_request_text_chars": self._last_request_metrics["text_chars"],
            "last_request_serialized_chars": self._last_request_metrics["serialized_chars"],
            "last_request_input_tokens": self._last_usage_metrics["input_tokens"],
            "last_request_output_tokens": self._last_usage_metrics["output_tokens"],
            "last_request_total_tokens": self._last_usage_metrics["total_tokens"],
            "last_request_cached_input_tokens": self._last_usage_metrics["cached_input_tokens"],
            "last_request_reasoning_output_tokens": self._last_usage_metrics["reasoning_output_tokens"],
            "cumulative_request_message_count": self._cumulative_request_metrics["message_count"],
            "cumulative_request_text_part_count": self._cumulative_request_metrics["text_part_count"],
            "cumulative_request_image_part_count": self._cumulative_request_metrics["image_part_count"],
            "cumulative_request_text_chars": self._cumulative_request_metrics["text_chars"],
            "cumulative_request_serialized_chars": self._cumulative_request_metrics["serialized_chars"],
            "cumulative_input_tokens": self._cumulative_usage_metrics["input_tokens"],
            "cumulative_output_tokens": self._cumulative_usage_metrics["output_tokens"],
            "cumulative_total_tokens": self._cumulative_usage_metrics["total_tokens"],
            "cumulative_cached_input_tokens": self._cumulative_usage_metrics["cached_input_tokens"],
            "cumulative_reasoning_output_tokens": self._cumulative_usage_metrics["reasoning_output_tokens"],
            **kwargs,
        }

    def _usage_snapshot(self) -> dict[str, dict[str, int]]:
        return {
            "last_request": {
                "message_count": self._last_request_metrics["message_count"],
                "text_part_count": self._last_request_metrics["text_part_count"],
                "image_part_count": self._last_request_metrics["image_part_count"],
                "input_tokens": self._last_usage_metrics["input_tokens"],
                "cached_input_tokens": self._last_usage_metrics["cached_input_tokens"],
            },
            "last_response": dict(self._last_usage_metrics),
            "cumulative_request": {
                "message_count": self._cumulative_request_metrics["message_count"],
                "text_part_count": self._cumulative_request_metrics["text_part_count"],
                "image_part_count": self._cumulative_request_metrics["image_part_count"],
                "input_tokens": self._cumulative_usage_metrics["input_tokens"],
                "cached_input_tokens": self._cumulative_usage_metrics["cached_input_tokens"],
            },
            "cumulative_response": dict(self._cumulative_usage_metrics),
        }

    def _log_gateway_error(self, *, event: str, attempt: int, error: BaseException) -> None:
        response = getattr(error, "response", None)
        response_text = ""
        if response is not None:
            try:
                response_text = str(getattr(response, "text", "") or "")
            except Exception:
                response_text = ""
        if len(response_text) > 4000:
            response_text = response_text[:4000]

        append_runtime_log(
            self.config.error_log_path,
            source="gateway",
            event=event,
            model_name=self.config.model_name,
            endpoint=self.config.openai_gateway_endpoint,
            attempt=attempt,
            error_type=type(error).__name__,
            error=str(error),
            status_code=getattr(response, "status_code", None),
            response_text=response_text,
        )

    def _raw_response_log_path(self) -> Path | None:
        if self.config.error_log_path is None:
            return None
        return self.config.error_log_path.parent / "raw_responses.jsonl"

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
            if self.config.attach_observation_screenshot and screenshot_path:
                parts.append(image_part_from_path(Path(screenshot_path)))

            observation_messages.append(
                self.format_message(role="user", content=parts, extra={"observation": observation})
            )
        return observation_messages

    def _response_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "bash_command": {"type": "string"},
                "python_code": {"type": "string"},
                "done": {"type": "boolean"},
                "final_response": {"type": "string"},
            },
            "required": ["thought", "done", "final_response"],
        }

    def _parse_model_output(self, raw_text: str) -> dict[str, Any]:
        if self.config.response_mode == "xml":
            return parse_xml_output(raw_text)
        parsed = parse_json_output(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Unable to parse JSON object from gateway response.")
        return _repair_done_with_command(parsed)

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

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        extra_body: dict[str, Any] = {"tier": self.config.openai_gateway_tier}
        if self.config.cache_ttl is not None:
            extra_body["cache_ttl"] = self.config.cache_ttl
        if self.config.session_id:
            extra_body["session_id"] = self.config.session_id
        if self.config.strict_session:
            extra_body["strict_session"] = True

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "input": _serialize_response_input(messages),
            "max_output_tokens": self.config.max_output_tokens,
            "extra_body": extra_body,
        }
        if self.config.response_mode == "json_schema":
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "playwright_step",
                    "schema": self._response_schema(),
                    "strict": False, #TODO check if this works -Yadong
                }
            }
        return payload

    def _format_repair_message(self, *, raw_text: str, error: str) -> dict[str, Any]:
        return self.format_message(
            role="user",
            content=Template(self.config.format_error_template, undefined=StrictUndefined).render(
                error=error,
                model_response=raw_text,
            ),
            extra={
                "interrupt_type": "FormatErrorRetry",
                "model_response": raw_text,
            },
        )

    def _sdk_base_url(self) -> str:
        base = self.config.openai_gateway_endpoint or ""
        if base.endswith("/responses"):
            base = base[: -len("/responses")]
        return base.rstrip("/")

    async def _query_async(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        base_url = self._sdk_base_url()

        last_error: ValueError | None = None
        raw_text = ""
        request_messages = list(messages)
        for attempt_index in range(MAX_JSON_PARSE_RETRIES + 1):
            payload = self._build_payload(request_messages)
            serialized_input = payload.get("input") or []
            request_metrics = _request_metrics_from_serialized_input(serialized_input)
            self._last_request_metrics = dict(request_metrics)
            for key, value in request_metrics.items():
                self._cumulative_request_metrics[key] += value
            response_payload = None
            for rate_limit_attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    async with AsyncOpenAI(
                        api_key=self.config.openai_gateway_api_key,
                        base_url=base_url,
                        timeout=self.config.request_timeout_seconds,
                    ) as client:
                        response = await client.responses.create(**payload)
                    response_payload = (
                        response.model_dump() if hasattr(response, "model_dump") else dict(response)
                    )
                    break
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        self._log_gateway_error(
                            event="rate_limit_error",
                            attempt=rate_limit_attempt + 1,
                            error=exc,
                        )
                        if rate_limit_attempt >= MAX_RATE_LIMIT_RETRIES:
                            raise
                        await asyncio.sleep(min(5 * (rate_limit_attempt + 1), 30))
                        continue
                    if _is_transient_gateway_error(exc):
                        self._log_gateway_error(
                            event="transient_gateway_error",
                            attempt=rate_limit_attempt + 1,
                            error=exc,
                        )
                        if rate_limit_attempt >= MAX_TRANSIENT_GATEWAY_RETRIES:
                            raise
                        await asyncio.sleep(min(2 * (rate_limit_attempt + 1), 10))
                        continue
                    self._log_gateway_error(
                        event="fatal_gateway_error",
                        attempt=rate_limit_attempt + 1,
                        error=exc,
                    )
                    raise

            if response_payload is None:
                raise RuntimeError("Gateway request returned no payload.")

            usage_metrics = _usage_metrics_from_response_payload(response_payload)
            self._last_usage_metrics = dict(usage_metrics)
            for key, value in usage_metrics.items():
                self._cumulative_usage_metrics[key] += value

            raw_text = _extract_response_text(response_payload)
            append_runtime_log(
                self._raw_response_log_path(),
                source="model",
                event="raw_text",
                attempt=attempt_index + 1,
                raw_text=raw_text,
            )
            try:
                parsed = self._parse_model_output(raw_text)
                break
            except ValueError as exc:
                last_error = exc
                if attempt_index < MAX_JSON_PARSE_RETRIES:
                    request_messages.append(
                        self._format_repair_message(raw_text=raw_text, error=str(exc))
                    )
        else:
            raise self._format_error(
                raw_text=raw_text,
                error=str(last_error or ValueError("Unable to parse model output.")),
            )

        actions = []
        bash_command = parsed.get("bash_command", "").strip()
        python_code = parsed.get("python_code", "").strip()
        if bash_command:
            try:
                _validate_bash_command(bash_command)
            except ValueError as exc:
                raise self._format_error(raw_text=raw_text, error=str(exc))
            actions.append({"bash_command": bash_command, "command": bash_command})
        elif python_code:
            actions.append({"python_code": python_code})

        return self.format_message(
            role="assistant",
            content=parsed.get("thought", ""),
            extra={
                "actions": actions,
                "done": bool(parsed.get("done", False)),
                "final_response": parsed.get("final_response", ""),
                "raw_response": parsed,
                "usage": self._usage_snapshot(),
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
                "usage": {
                    **self._usage_snapshot(),
                },
                "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            }
        }
