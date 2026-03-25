from __future__ import annotations

import pytest

from miniswewebagent.exceptions import FormatError
from miniswewebagent.models.phyagi_model import PhyagiModel, parse_xml_output


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
                "choices": [
                    {
                        "message": {
                            "content": """
<response>
  <thought>Next step</thought>
  <python_code><![CDATA[
await page.title()
]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
                            """.strip()
                        }
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
            assert "response_format" not in json
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
            return {"choices": [{"message": {"content": "not valid xml"}}]}

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
