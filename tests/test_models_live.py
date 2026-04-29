"""Live integration test for OpenAIModel + AnthropicModel.

Sources credentials from ~/cred.sh (or $CRED_FILE) and makes one real API call
against each backend. Run manually:

    bash Webwright/tests/test_models_live.sh

or directly:

    source ~/cred.sh && python -m pytest Webwright/tests/test_models_live.py -s

Skips silently when an API key is not present.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from webwright.models.anthropic_model import AnthropicModel
from webwright.models.openai_model import OpenAIModel


PROMPT = (
    "Say hi by emitting a strict JSON object with: "
    'thought="hello", bash_command="", done=true, final_response="hi".'
)


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a test agent. Reply with JSON only."},
        {"role": "user", "content": PROMPT, "extra": {}},
    ]


def _print_response(label: str, response: dict[str, Any]) -> None:
    extra = response["extra"]
    print(f"\n=== [{label}] full parsed response ===")
    print(json.dumps(extra["raw_response"], indent=2))
    print(f"=== [{label}] assistant.content (thought) ===")
    print(response["content"])
    print(f"=== [{label}] actions ===")
    print(json.dumps(extra["actions"], indent=2))
    print(f"=== [{label}] done={extra['done']} final_response={extra['final_response']!r} ===")
    print(f"=== [{label}] usage ===")
    print(json.dumps(extra["usage"]["last_response"], indent=2))


def _assert_response_shape(response: dict[str, Any]) -> None:
    assert response["role"] == "assistant"
    extra = response["extra"]
    parsed = extra["raw_response"]
    assert isinstance(parsed, dict)
    for key in ("thought", "bash_command", "done", "final_response"):
        assert key in parsed, f"missing {key!r} in {parsed!r}"
    assert isinstance(parsed["done"], bool)
    usage = extra["usage"]["last_response"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_openai_model_live() -> None:
    model = OpenAIModel(model_name=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4"))
    response = model.query(_messages())
    _print_response("openai", response)
    _assert_response_shape(response)


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_model_live() -> None:
    model = AnthropicModel(
        model_name=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-opus-4-7"),
    )
    response = model.query(_messages())
    _print_response("anthropic", response)
    _assert_response_shape(response)
