from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from miniswewebagent.agents.default import DefaultAgent
from miniswewebagent.tools import image_qa, self_reflection
from miniswewebagent.utils.judge_gateway import (
    ensure_policy_only_not_bypassed,
    resolve_policy_judge,
)

POLICY_ENVS = (
    "OPENAI_GATEWAY_MODEL",
    "OPENAI_GATEWAY_ENDPOINT",
    "OPENAI_COMPATIBLE_ENDPOINT",
    "OPENAI_COMPATIBLE_MODEL",
    "OPENAI_COMPATIBLE_API_KEY",
    "WEB_AGENT_POLICY_URL",
    "WEB_AGENT_POLICY_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_judge_env(monkeypatch) -> None:
    for name in POLICY_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "configured, expected",
    [
        ("http://127.0.0.1:8002", "http://127.0.0.1:8002/v1/chat/completions"),
        ("http://127.0.0.1:8002/v1", "http://127.0.0.1:8002/v1/chat/completions"),
        ("http://127.0.0.1:8002/v1/", "http://127.0.0.1:8002/v1/chat/completions"),
        (
            "http://127.0.0.1:8002/v1/chat/completions",
            "http://127.0.0.1:8002/v1/chat/completions",
        ),
    ],
)
def test_resolve_policy_judge_normalizes_endpoint(monkeypatch, configured, expected) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_ENDPOINT", configured)
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "sft_ckpt")

    target = resolve_policy_judge(tool="test")

    assert (target.endpoint, target.model, target.api_key) == (expected, "sft_ckpt", "dummy")


def test_resolve_policy_judge_falls_back_to_web_agent_policy_env(monkeypatch) -> None:
    monkeypatch.setenv("WEB_AGENT_POLICY_URL", "http://127.0.0.1:8003/v1/chat/completions")
    monkeypatch.setenv("WEB_AGENT_POLICY_MODEL", "sft_ckpt")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-local")

    target = resolve_policy_judge(tool="test")

    assert target.endpoint == "http://127.0.0.1:8003/v1/chat/completions"
    assert target.model == "sft_ckpt"
    assert target.api_key == "sk-local"


def test_resolve_policy_judge_rejects_responses_gateway(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_ENDPOINT", "http://gateway.phyagi.net/api/responses")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "sft_ckpt")

    with pytest.raises(RuntimeError, match="/responses gateway URL"):
        resolve_policy_judge(tool="test")


def test_resolve_policy_judge_requires_endpoint_and_model(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="no policy endpoint is configured"):
        resolve_policy_judge(tool="test")

    monkeypatch.setenv("OPENAI_COMPATIBLE_ENDPOINT", "http://127.0.0.1:8002/v1")
    with pytest.raises(RuntimeError, match="served model name is unknown"):
        resolve_policy_judge(tool="test")


def test_self_reflection_gateway_config_uses_policy_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_MODEL", "policy")
    monkeypatch.setenv("OPENAI_GATEWAY_ENDPOINT", "http://gateway.phyagi.net/api/responses")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "sk-gateway")
    monkeypatch.setenv("OPENAI_COMPATIBLE_ENDPOINT", "http://127.0.0.1:8002/v1/chat/completions")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "sft_ckpt")

    config = self_reflection._gateway_config(api_key="", endpoint="", model="")

    assert config.backend == "policy_chat"
    assert config.endpoint == "http://127.0.0.1:8002/v1/chat/completions"
    assert config.model == "sft_ckpt"


def test_self_reflection_explicit_model_still_reaches_the_gateway(monkeypatch) -> None:
    """--model wins over the policy sentinel in the environment."""
    monkeypatch.setenv("OPENAI_GATEWAY_MODEL", "policy")
    monkeypatch.setenv("OPENAI_GATEWAY_ENDPOINT", "http://gateway.phyagi.net/api/responses")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "sk-gateway")

    config = self_reflection._gateway_config(api_key="", endpoint="", model="gpt-5.4")

    assert config.backend == "responses"
    assert config.endpoint == "http://gateway.phyagi.net/api/responses"
    assert config.model == "gpt-5.4"


def test_self_reflection_policy_call_posts_chat_completions(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_MODEL", "policy")
    monkeypatch.setenv("OPENAI_COMPATIBLE_ENDPOINT", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "sft_ckpt")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-local")

    seen: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": "Reasoning: looks right\nScore: 4"}}
                ]
            }

    class FakeClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            seen["url"] = url
            seen["headers"] = headers
            seen["json"] = json
            return FakeResponse()

    monkeypatch.setattr(self_reflection.httpx, "Client", FakeClient)

    config = self_reflection._gateway_config(api_key="", endpoint="", model="")
    text = self_reflection._call_gateway(
        system_prompt="judge system",
        user_content=[
            {"type": "input_text", "text": "judge user"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "high"},
        ],
        gateway_config=config,
        timeout_seconds=30,
        max_new_tokens=256,
        max_attempts=1,
        retry_base_delay=0.0,
        tag="test",
    )

    assert text == "Reasoning: looks right\nScore: 4"
    assert seen["url"] == "http://127.0.0.1:8002/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-local"
    assert seen["json"]["model"] == "sft_ckpt"
    assert seen["json"]["messages"][0] == {"role": "system", "content": "judge system"}
    user_parts = seen["json"]["messages"][1]["content"]
    assert user_parts[0] == {"type": "text", "text": "judge user"}
    assert user_parts[1]["type"] == "image_url"
    assert user_parts[1]["image_url"]["url"] == "data:image/png;base64,AA"


def test_image_qa_policy_posts_chat_completions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_MODEL", "policy")
    monkeypatch.setenv("OPENAI_COMPATIBLE_ENDPOINT", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "sft_ckpt")

    api_key, endpoint, model = image_qa._gateway_config(
        argparse.Namespace(endpoint="", model="", api_key="")
    )
    assert (api_key, endpoint, model) == (
        "dummy",
        "http://127.0.0.1:8002/v1/chat/completions",
        "sft_ckpt",
    )

    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    seen: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"answer": "blue", "evidence": ["a blue chip"], '
                                '"unknown": false, "confidence": 0.9}'
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            seen["url"] = url
            seen["json"] = json
            return FakeResponse()

    monkeypatch.setattr(image_qa.httpx, "Client", FakeClient)

    result = image_qa.run_image_qa(
        image_path=image_path,
        question="What colour is the chip?",
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        timeout_seconds=30,
    )

    assert result["answer"] == "blue"
    assert seen["url"] == "http://127.0.0.1:8002/v1/chat/completions"
    assert seen["json"]["model"] == "sft_ckpt"
    assert seen["json"]["response_format"]["type"] == "json_schema"
    assert seen["json"]["messages"][0]["content"][1]["type"] == "image_url"


class _StubModelConfig:
    endpoint = "http://127.0.0.1:8003/v1/chat/completions"
    model_name = "sft_ckpt"
    api_key = "dummy"


class _StubPolicyModel:
    config = _StubModelConfig()

    def get_template_vars(self, **kwargs):
        return kwargs


class _StubEnvConfig:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env


class _StubEnvironment:
    def __init__(self, env: dict[str, str]) -> None:
        self.config = _StubEnvConfig(env)

    def get_template_vars(self, **kwargs):
        return kwargs


def _agent_with_judge_model(judge_model: str, env: dict[str, str]) -> DefaultAgent:
    return DefaultAgent(
        _StubPolicyModel(),
        _StubEnvironment(env),
        system_template="system",
        instance_template="instance",
        judge_mode="trajectory",
        judge_model=judge_model,
    )


def test_agent_policy_judge_mirrors_model_endpoint_into_workspace_env() -> None:
    env = {
        "OPENAI_GATEWAY_ENDPOINT": "http://gateway.phyagi.net/api/responses",
        "OPENAI_GATEWAY_MODEL": "gpt-5.4",
        # Stale hardcoded port from the config; model.endpoint must win.
        "OPENAI_COMPATIBLE_ENDPOINT": "http://127.0.0.1:8000/v1/chat/completions",
        "OPENAI_COMPATIBLE_MODEL": "sft_ckpt",
    }

    _agent_with_judge_model("policy", env)

    assert env["OPENAI_GATEWAY_MODEL"] == "policy"
    assert env["OPENAI_COMPATIBLE_ENDPOINT"] == "http://127.0.0.1:8003/v1/chat/completions"
    assert env["OPENAI_COMPATIBLE_MODEL"] == "sft_ckpt"
    assert env["OPENAI_COMPATIBLE_API_KEY"] == "dummy"
    # Policy-only: the judge gateway is removed and the guard flag is set, so no
    # judge call can silently reroute off the vLLM server under evaluation.
    assert "OPENAI_GATEWAY_ENDPOINT" not in env
    assert env["MWA_JUDGE_POLICY_ONLY"] == "1"


def test_policy_only_run_rejects_a_non_policy_judge_backend(monkeypatch) -> None:
    monkeypatch.setenv("MWA_JUDGE_POLICY_ONLY", "1")
    with pytest.raises(RuntimeError, match="policy-only"):
        ensure_policy_only_not_bypassed(model="gpt-5.4", tool="self_reflection")


def test_policy_only_guard_is_inert_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("MWA_JUDGE_POLICY_ONLY", raising=False)
    ensure_policy_only_not_bypassed(model="gpt-5.4", tool="self_reflection")


def test_agent_leaves_env_untouched_for_a_regular_judge_model() -> None:
    env = {
        "OPENAI_GATEWAY_ENDPOINT": "http://gateway.phyagi.net/api/responses",
        "OPENAI_GATEWAY_MODEL": "gpt-5.4",
    }

    _agent_with_judge_model("o4-mini", env)

    assert env == {
        "OPENAI_GATEWAY_ENDPOINT": "http://gateway.phyagi.net/api/responses",
        "OPENAI_GATEWAY_MODEL": "gpt-5.4",
    }
