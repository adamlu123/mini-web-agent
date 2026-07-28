from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import httpx

from miniswewebagent.config import get_config_from_spec
from miniswewebagent.models import get_model
from miniswewebagent.models.openai_compatible_model import (
    OpenAICompatibleModel,
    _normalize_chat_completions_endpoint,
)


def test_registry_resolves_vllm_alias() -> None:
    model = get_model(
        {
            "model_class": "vllm",
            "model_name": "qwen35_9b_base",
            "endpoint": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
        }
    )

    assert isinstance(model, OpenAICompatibleModel)
    assert model.config.endpoint == "http://127.0.0.1:8000/v1/chat/completions"


def test_endpoint_normalization() -> None:
    assert (
        _normalize_chat_completions_endpoint("http://localhost:8000")
        == "http://localhost:8000/v1/chat/completions"
    )
    assert (
        _normalize_chat_completions_endpoint("http://localhost:8000/v1/")
        == "http://localhost:8000/v1/chat/completions"
    )
    assert (
        _normalize_chat_completions_endpoint(
            "http://localhost:8000/v1/chat/completions"
        )
        == "http://localhost:8000/v1/chat/completions"
    )


def test_qwen35_vllm_config_resolves() -> None:
    config = get_config_from_spec("model_vllm_9b_base.yaml")

    assert config["model"]["model_class"] == "vllm"
    assert config["model"]["model_name"] == "qwen35_9b_base"
    assert config["model"]["response_mode"] == "json_schema"
    assert config["model"]["max_context_tokens"] > 0


def test_openai_compatible_query_with_mocked_http(monkeypatch) -> None:
    model = OpenAICompatibleModel(
        model_name="qwen35_9b_base",
        endpoint="http://127.0.0.1:8000/v1",
        api_key="dummy",
    )
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "thought": "inspect the page",
                                    "bash_command": "echo hi",
                                    "done": False,
                                    "final_response": "",
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = model.query([{"role": "user", "content": "task"}])

    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["payload"]["model"] == "qwen35_9b_base"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "task"}]
    assert result["extra"]["actions"] == [{"bash_command": "echo hi", "command": "echo hi"}]
    assert result["extra"]["raw_text"]
    assert model._last_usage_metrics["total_tokens"] == 18


def test_sliding_window_preserves_task_and_recent_turn() -> None:
    model = OpenAICompatibleModel(
        model_name="qwen35_9b_base",
        api_key="dummy",
        max_context_tokens=500,
        sliding_window_keep_turns=1,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        {
            "role": "assistant",
            "content": "turn one",
            "extra": {
                "raw_response": {
                    "thought": "turn one " + ("x" * 800),
                    "bash_command": "echo one",
                    "done": False,
                    "final_response": "",
                }
            },
        },
        {"role": "user", "content": "observation one"},
        {
            "role": "assistant",
            "content": "turn two",
            "extra": {
                "raw_response": {
                    "thought": "turn two " + ("y" * 800),
                    "bash_command": "echo two",
                    "done": False,
                    "final_response": "",
                }
            },
        },
        {"role": "user", "content": "observation two"},
    ]

    payload = model._build_payload(messages)
    serialized = json.dumps(payload["messages"])

    assert payload["messages"][0]["content"] == "system"
    assert payload["messages"][1]["content"] == "original task"
    assert payload["messages"][2]["content"].startswith("[Context truncated:")
    assert "turn one" not in serialized
    assert "turn two" in serialized


def test_vllm_launcher_can_use_existing_server(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ | {
        "START_VLLM": "0",
        "EVALUATE": "0",
        "PY": "/bin/echo",
        "CREDENTIALS_FILE": str(tmp_path / "missing-credentials.sh"),
        "TASKS_FILE": str(tmp_path / "tasks.json"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "ENDPOINT": "http://inference.example:9000/v1/chat/completions",
        "OPENAI_GATEWAY_ENDPOINT": "http://judge.example/api/responses",
    }

    completed = subprocess.run(
        ["bash", str(repository / "scripts" / "run_vllm_qwen35_om2w.sh")],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "-c best_default_judge_json_agnostic.yaml" in completed.stdout
    assert "-c model_vllm_9b_base.yaml" in completed.stdout
    assert "model.endpoint=http://inference.example:9000/v1/chat/completions" in completed.stdout
    assert "--judge-endpoint http://judge.example/api/responses" in completed.stdout
    assert "--no-evaluate" in completed.stdout
