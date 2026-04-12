from __future__ import annotations

from pathlib import Path

import pytest
import httpx

from miniswewebagent.exceptions import FormatError
from miniswewebagent.models.phyagi_model import (
    PhyagiModel,
    _serialize_response_input,
    parse_xml_output,
)


def test_parse_xml_output_with_cdata() -> None:
    parsed = parse_xml_output(
        """
<response>
  <thought>Tried: open form. Failed because: no execution error. Next step: fill route.</thought>
  <python_code><![CDATA[
await page.get_by_role("textbox", name="From").fill("Singapore")
]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip()
    )

    assert parsed["thought"].startswith("Tried:")
    assert parsed["bash_command"] == ""
    assert 'fill("Singapore")' in parsed["python_code"]
    assert parsed["done"] is False
    assert parsed["final_response"] == ""


def test_parse_xml_output_accepts_bash_command() -> None:
    parsed = parse_xml_output(
        """
<response>
  <thought>Inspect the workspace.</thought>
  <bash_command><![CDATA[
echo hello
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip()
    )

    assert parsed["thought"] == "Inspect the workspace."
    assert parsed["bash_command"] == "echo hello"
    assert parsed["python_code"] == ""
    assert parsed["done"] is False
    assert parsed["final_response"] == ""


def test_phyagi_model_formats_observations_from_template() -> None:
    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        observation_template="Status={{ 'ok' if observation.success else 'error' }} URL={{ observation.url }}",
    )

    messages = model.format_observation_messages(
        {},
        [
            {
                "observation": {
                    "success": True,
                    "url": "https://example.com",
                    "title": "Example",
                    "screenshot_path": "",
                }
            }
        ],
    )

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "Status=ok URL=https://example.com"


def test_phyagi_model_attaches_observation_screenshot_by_default(tmp_path: Path) -> None:
    screenshot = tmp_path / "shot.png"
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\n")

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        observation_template="Status={{ 'ok' if observation.success else 'error' }}",
    )

    messages = model.format_observation_messages(
        {},
        [
            {
                "observation": {
                    "success": True,
                    "url": "https://example.com",
                    "title": "Example",
                    "screenshot_path": str(screenshot),
                }
            }
        ],
    )

    assert [part["type"] for part in messages[0]["content"]] == ["input_text", "input_image"]


def test_phyagi_model_can_disable_observation_screenshot_attachment(tmp_path: Path) -> None:
    screenshot = tmp_path / "shot.png"
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\n")

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        observation_template="Status={{ 'ok' if observation.success else 'error' }}",
        attach_observation_screenshot=False,
    )

    messages = model.format_observation_messages(
        {},
        [
            {
                "observation": {
                    "success": True,
                    "url": "https://example.com",
                    "title": "Example",
                    "screenshot_path": str(screenshot),
                }
            }
        ],
    )

    assert [part["type"] for part in messages[0]["content"]] == ["input_text"]


def test_phyagi_model_query_parses_xml_mode(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": """
<response>
  <thought>Next step</thought>
  <python_code><![CDATA[
await page.title()
]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
                                """.strip(),
                            }
                        ],
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            assert "messages" not in json
            assert "input" in json
            assert "text" not in json
            assert json["max_output_tokens"] == 4000
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.httpx.AsyncClient", FakeAsyncClient)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="error: {{ error }}",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert message["content"] == "Next step"
    assert message["extra"]["actions"] == [{"python_code": "await page.title()"}]
    assert message["extra"]["done"] is False


def test_phyagi_model_query_parses_xml_mode_with_bash_command(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": """
<response>
  <thought>Inspect files</thought>
  <bash_command><![CDATA[
pwd
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
                                """.strip(),
                            }
                        ],
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.httpx.AsyncClient", FakeAsyncClient)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="error: {{ error }}",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert message["content"] == "Inspect files"
    assert message["extra"]["actions"] == [{"bash_command": "pwd", "command": "pwd"}]
    assert message["extra"]["done"] is False
    assert message["extra"]["usage"]["last_response"]["input_tokens"] == 0


def test_phyagi_model_query_raises_format_error_for_bad_xml(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "not valid xml"}],
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.httpx.AsyncClient", FakeAsyncClient)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="error: {{ error }}",
    )

    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "Task"}])


def test_phyagi_model_query_repairs_plain_text_with_format_retry(monkeypatch) -> None:
    attempts = {"count": 0}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            attempts["count"] += 1
            if attempts["count"] == 1:
                text = "Need more evidence before clicking Filters."
            else:
                text = """
<response>
  <thought>Retry worked</thought>
  <bash_command><![CDATA[
ls
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
                """.strip()
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.httpx.AsyncClient", FakeAsyncClient)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="Repair XML: {{ error }}",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert attempts["count"] == 2
    assert message["content"] == "Retry worked"
    assert message["extra"]["actions"] == [{"bash_command": "ls", "command": "ls"}]


def test_phyagi_model_query_retries_transient_gateway_errors(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": """
<response>
  <thought>Retry worked</thought>
  <python_code><![CDATA[
]]></python_code>
  <done>true</done>
  <final_response>ok</final_response>
</response>
                                """.strip(),
                            }
                        ],
                    }
                ]
            }

    attempts = {"count": 0}

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers: dict, json: dict):
            attempts["count"] += 1
            if attempts["count"] < 3:
                request = httpx.Request("POST", url)
                response = httpx.Response(502, request=request)
                raise httpx.HTTPStatusError("502 Bad Gateway", request=request, response=response)
            return FakeResponse()

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.asyncio.sleep", fake_sleep)

    model = PhyagiModel(openai_gateway_api_key="dummy", response_mode="xml")
    message = model.query([{"role": "user", "content": "Task"}])

    assert attempts["count"] == 3
    assert message["extra"]["done"] is True
    assert message["extra"]["final_response"] == "ok"


def test_phyagi_model_query_exposes_gateway_usage_in_template_vars(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "usage": {
                    "input_tokens": 123,
                    "output_tokens": 45,
                    "total_tokens": 168,
                    "input_tokens_details": {"cached_tokens": 7},
                    "output_tokens_details": {"reasoning_tokens": 11},
                },
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": """
<response>
  <thought>Inspect files</thought>
  <bash_command><![CDATA[
pwd
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
                                """.strip(),
                            }
                        ],
                    }
                ],
            }

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.httpx.AsyncClient", FakeAsyncClient)

    model = PhyagiModel(openai_gateway_api_key="dummy", response_mode="xml")
    model.query([{"role": "user", "content": "Task"}])

    template_vars = model.get_template_vars()

    assert template_vars["last_request_message_count"] == 1
    assert template_vars["last_request_text_chars"] == 4
    assert template_vars["last_request_input_tokens"] == 123
    assert template_vars["last_request_output_tokens"] == 45
    assert template_vars["last_request_total_tokens"] == 168
    assert template_vars["last_request_cached_input_tokens"] == 7
    assert template_vars["last_request_reasoning_output_tokens"] == 11
    assert template_vars["cumulative_input_tokens"] == 123
    assert template_vars["cumulative_output_tokens"] == 45
    usage_snapshot = model._usage_snapshot()
    assert usage_snapshot["last_request"]["input_tokens"] == 123
    assert usage_snapshot["last_request"]["cached_input_tokens"] == 7
    assert "text_chars" not in usage_snapshot["last_request"]
    assert usage_snapshot["cumulative_request"]["input_tokens"] == 123
    assert usage_snapshot["last_response"]["input_tokens"] == 123


def test_phyagi_model_query_uses_zero_usage_when_gateway_omits_usage(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": """
<response>
  <thought>Inspect files</thought>
  <bash_command><![CDATA[
pwd
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
                                """.strip(),
                            }
                        ],
                    }
                ],
            }

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.httpx.AsyncClient", FakeAsyncClient)

    model = PhyagiModel(openai_gateway_api_key="dummy", response_mode="xml")
    model.query([{"role": "user", "content": "Task"}])

    template_vars = model.get_template_vars()

    assert template_vars["last_request_input_tokens"] == 0
    assert template_vars["cumulative_input_tokens"] == 0
    assert template_vars["last_request_text_chars"] == 4
    assert template_vars["cumulative_request_text_chars"] == 4
    usage_snapshot = model._usage_snapshot()
    assert usage_snapshot["last_request"]["input_tokens"] == 0
    assert usage_snapshot["cumulative_request"]["input_tokens"] == 0
    assert "text_chars" not in usage_snapshot["last_request"]


def test_serialize_response_input_uses_output_text_for_assistant_messages() -> None:
    serialized = _serialize_response_input(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ]
    )

    assert serialized[0]["role"] == "developer"
    assert serialized[0]["content"][0]["type"] == "input_text"
    assert serialized[1]["role"] == "user"
    assert serialized[1]["content"][0]["type"] == "input_text"
    assert serialized[2]["role"] == "assistant"
    assert serialized[2]["content"][0]["type"] == "output_text"
