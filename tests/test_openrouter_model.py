"""Tests for OpenRouterModel.

Includes:
- Unit tests using mocked HTTP (always run).
- A live endpoint smoke test (skipped unless OPENROUTER_API_KEY is set and
  OPENROUTER_LIVE_TEST=1 is exported — opt-in to avoid spending credits on CI).

Usage for the live test:
    source /home/luyadong/cred.sh
    OPENROUTER_LIVE_TEST=1 pytest -q tests/test_openrouter_model.py -k live -s
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from miniswewebagent.models import get_model
from miniswewebagent.models.openrouter_model import (
    OpenRouterModel,
    _extract_chat_text,
    _serialize_chat_messages,
    _usage_from_chat_payload,
)


def test_registry_resolves_openrouter_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    model = get_model({"model_class": "openrouter"})
    assert isinstance(model, OpenRouterModel)
    assert model.config.model_name == "qwen/qwen3.5-27b"


def test_serialize_chat_messages_collapses_text_parts() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "caption"}, {"type": "input_image", "image_url": "data:image/png;base64,xxx"}]},
        {"role": "exit", "content": "drop me"},
    ]
    out = _serialize_chat_messages(messages)
    assert len(out) == 3
    assert out[0] == {"role": "user", "content": "hello"}
    assert out[1] == {"role": "user", "content": "a\nb"}
    assert isinstance(out[2]["content"], list)
    assert out[2]["content"][0] == {"type": "text", "text": "caption"}
    assert out[2]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}


def test_extract_chat_text_and_usage() -> None:
    payload = {
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    assert _extract_chat_text(payload) == "hello world"
    usage = _usage_from_chat_payload(payload)
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 12


def test_query_with_mocked_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    model = OpenRouterModel()

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def __init__(self, data: dict) -> None:
            self._data = data
            self.text = json.dumps(data)

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._data

    class FakeClient:
        def __init__(self, *a, **kw) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<response>\n"
                                    "  <thought>ok</thought>\n"
                                    "  <bash_command><![CDATA[echo hi]]></bash_command>\n"
                                    "  <done>false</done>\n"
                                    "  <final_response></final_response>\n"
                                    "</response>"
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = model.query([{"role": "user", "content": "hi"}])
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "qwen/qwen3.5-27b"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result["role"] == "assistant"
    assert result["extra"]["actions"] == [{"bash_command": "echo hi", "command": "echo hi"}]
    assert model._last_usage_metrics["input_tokens"] == 3
    assert model._last_usage_metrics["output_tokens"] == 4


@pytest.mark.skipif(
    not (os.environ.get("OPENROUTER_API_KEY") and os.environ.get("OPENROUTER_LIVE_TEST")),
    reason="Set OPENROUTER_API_KEY and OPENROUTER_LIVE_TEST=1 to run the live endpoint test.",
)
def test_live_openrouter_endpoint() -> None:
    """Hit the real OpenRouter endpoint with a tiny prompt."""
    model = OpenRouterModel(
        model_name=os.environ.get("OPENROUTER_TEST_MODEL", "qwen/qwen3.5-27b"),
        max_output_tokens=200,
    )
    result = model.query(
        [
            {
                "role": "system",
                "content": (
                    "Respond ONLY with this XML and nothing else:\n"
                    "<response><thought>ping</thought>"
                    "<bash_command><![CDATA[echo pong]]></bash_command>"
                    "<done>false</done><final_response></final_response></response>"
                ),
            },
            {"role": "user", "content": "Please produce the XML as instructed."},
        ]
    )
    assert result["role"] == "assistant"
    # Usage telemetry should be populated by a real call.
    assert model._last_usage_metrics["input_tokens"] > 0
    assert model._last_usage_metrics["output_tokens"] > 0
    print("\n[live] thought:", result["content"])
    print("[live] actions:", result["extra"]["actions"])
    print("[live] usage:", model._last_usage_metrics)
