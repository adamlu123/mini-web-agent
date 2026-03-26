from __future__ import annotations

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
    assert 'fill("Singapore")' in parsed["python_code"]
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
