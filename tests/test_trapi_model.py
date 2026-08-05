"""Tests for TrapiModel and the batch runner's per-task endpoint assignment.

Includes:
- Unit tests for the payload shape, schema strictness, and rate-limit backoff.
- Unit tests for uniform endpoint distribution / per-task recording.
- A live TRAPI smoke test (skipped unless TRAPI_LIVE_TEST=1 is exported and the
  host has an Azure CLI login that can mint an `api://trapi/.default` token).

Usage for the live test:
    TRAPI_LIVE_TEST=1 pytest -q tests/test_trapi_model.py -k live -s
"""

from __future__ import annotations

import json
import os
import types

import pytest

from miniswewebagent.models import get_model
from miniswewebagent.models.trapi_kimi_model import TrapiKimiModel, TrapiKimiModelConfig
from miniswewebagent.models.trapi_model import (
    DEFAULT_TRAPI_DEPLOYMENTS,
    TrapiModel,
    TrapiModelConfig,
    _retry_hint_from_body,
)
from miniswewebagent.run.benchmarks.om2w import (
    _assign_model_endpoints,
    _record_task_endpoint,
    _select_tasks,
)


def _model(**overrides) -> TrapiModel:
    model = TrapiModel.__new__(TrapiModel)
    model.config = TrapiModelConfig(**overrides)
    return model


def _rate_limit_exc(body: str, retry_after: str | None = None) -> Exception:
    exc = Exception("429 Too Many Requests")
    exc.response = types.SimpleNamespace(
        text=body,
        headers={"retry-after": retry_after} if retry_after else {},
        status_code=429,
    )
    return exc


def test_registry_resolves_trapi_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TrapiModel, "_build_token_provider", lambda self: (lambda: "token"))
    model = get_model({"model_class": "trapi"})
    assert isinstance(model, TrapiModel)
    assert model.config.model_name == "gpt-5.4_2026-03-05"


def test_chat_completions_url_targets_the_deployment() -> None:
    model = _model(model_name="gpt-5.6-sol_2026-07-09")
    assert model._chat_completions_url() == (
        "https://trapi.research.microsoft.com/gcr/shared/openai/deployments/"
        "gpt-5.6-sol_2026-07-09/chat/completions?api-version=2025-04-01-preview"
    )


def test_payload_uses_max_completion_tokens_not_max_tokens() -> None:
    """gpt-5.x rejects `max_tokens` with an `unsupported_parameter` 400."""
    payload = _model(max_output_tokens=16000)._build_payload([{"role": "user", "content": "hi"}])
    assert payload["max_completion_tokens"] == 16000
    assert "max_tokens" not in payload


def test_payload_includes_reasoning_effort_only_when_set() -> None:
    assert "reasoning_effort" not in _model()._build_payload([{"role": "user", "content": "hi"}])
    payload = _model(reasoning_effort="xhigh")._build_payload([{"role": "user", "content": "hi"}])
    assert payload["reasoning_effort"] == "xhigh"


def test_blank_reasoning_effort_is_treated_as_unset() -> None:
    assert TrapiModelConfig(reasoning_effort="").reasoning_effort is None


def test_json_schema_is_non_strict_by_default() -> None:
    """Strict mode requires `additionalProperties: false` plus all-required."""
    payload = _model(response_mode="json_schema")._build_payload([{"role": "user", "content": "hi"}])
    assert payload["response_format"]["json_schema"]["strict"] is False
    payload = _model(response_mode="json_object")._build_payload([{"role": "user", "content": "hi"}])
    assert payload["response_format"] == {"type": "json_object"}
    assert "response_format" not in _model(response_mode="none")._build_payload([])


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"statusCode": 429, "message": "Token limit is exceeded. Try again in 6 seconds."}', 6.0),
        ('{"status": 429, "error": "TRAPI: Rate Limit Exceeded, retry after 1 seconds."}', 1.0),
        ('{"statusCode": 429, "message": "Too many requests"}', None),
    ],
)
def test_retry_hint_parsed_from_429_body(body: str, expected: float | None) -> None:
    assert _retry_hint_from_body(_rate_limit_exc(body)) == expected


def test_rate_limit_delay_honors_the_hint() -> None:
    model = _model()
    exc = _rate_limit_exc('{"message": "Token limit is exceeded. Try again in 6 seconds."}')
    delays = [model._rate_limit_delay(exc, 0) for _ in range(200)]
    assert min(delays) >= 6.0
    assert max(delays) <= 6.0 + model.config.rate_limit_jitter_seconds


def test_rate_limit_delay_backs_off_exponentially_without_a_hint() -> None:
    model = _model()
    exc = _rate_limit_exc('{"message": "Too many requests"}')
    assert model._rate_limit_delay(exc, 0) < model._rate_limit_delay(exc, 5)
    assert model._rate_limit_delay(exc, 20) <= model.config.rate_limit_max_backoff_seconds


def test_rate_limit_delay_is_floored_and_capped() -> None:
    model = _model()
    tiny = model._rate_limit_delay(_rate_limit_exc('{"message": "try again in 0 seconds"}'), 0)
    assert tiny >= model.config.rate_limit_min_backoff_seconds
    huge = model._rate_limit_delay(_rate_limit_exc('{"message": "try again in 9999 seconds"}'), 0)
    assert huge <= model.config.rate_limit_max_backoff_seconds


def test_kimi_backoff_is_unchanged_by_the_trapi_override() -> None:
    """Kimi's pool stays saturated for long windows; keep its 30-60s desync wait."""
    model = TrapiKimiModel.__new__(TrapiKimiModel)
    model.config = TrapiKimiModelConfig()
    exc = _rate_limit_exc('{"message": "Token limit is exceeded. Try again in 6 seconds."}')
    delays = [model._rate_limit_delay(exc, 0) for _ in range(200)]
    assert 30.0 <= min(delays) and max(delays) <= 60.0


def test_endpoints_are_distributed_uniformly_and_pinned_per_task() -> None:
    tasks = [{"task_id": f"t{i}"} for i in range(500)]
    assignments = _assign_model_endpoints(tasks, list(DEFAULT_TRAPI_DEPLOYMENTS))
    assert len(assignments) == len(tasks)
    counts = {ep: list(assignments.values()).count(ep) for ep in DEFAULT_TRAPI_DEPLOYMENTS}
    assert set(counts.values()) == {125}


def test_endpoint_distribution_stays_balanced_when_not_divisible() -> None:
    tasks = [{"task_id": f"t{i}"} for i in range(10)]
    counts = list(
        {
            ep: list(_assign_model_endpoints(tasks, ["a", "b", "c", "d"]).values()).count(ep)
            for ep in ("a", "b", "c", "d")
        }.values()
    )
    assert max(counts) - min(counts) <= 1


def test_no_endpoints_means_no_assignment() -> None:
    assert _assign_model_endpoints([{"task_id": "t0"}], []) == {}


def test_record_task_endpoint_writes_the_assignment(tmp_path) -> None:
    _record_task_endpoint(tmp_path / "t0", "t0", "gpt-5.5_2026-04-24")
    payload = json.loads((tmp_path / "t0" / "model_endpoint.json").read_text())
    assert payload["task_id"] == "t0"
    assert payload["model_endpoint"] == "gpt-5.5_2026-04-24"
    assert payload["assigned_at"]


def test_select_tasks_offset_and_limit_pick_a_row_range(tmp_path) -> None:
    tasks_file = tmp_path / "om2w.json"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": f"t{i}",
                    "confirmed_task": f"Task {i}",
                    "website": f"https://example.com/{i}",
                    "level": "hard",
                }
                for i in range(10)
            ]
        ),
        encoding="utf-8",
    )
    selected = _select_tasks(tasks_file, [], 3, "hard", offset=5)
    assert [task["task_id"] for task in selected] == ["t5", "t6", "t7"]
    assert len(_select_tasks(tasks_file, [], 0, "hard", offset=8)) == 2
    assert len(_select_tasks(tasks_file, [], 0, "hard")) == 10


@pytest.mark.skipif(
    os.environ.get("TRAPI_LIVE_TEST") != "1",
    reason="Set TRAPI_LIVE_TEST=1 (with an Azure CLI login) to hit the real deployments.",
)
@pytest.mark.parametrize("deployment", DEFAULT_TRAPI_DEPLOYMENTS)
def test_live_deployment_answers_with_schema_json(deployment: str) -> None:
    import asyncio

    model = get_model(
        {
            "model_class": "trapi",
            "model_name": deployment,
            "response_mode": "json_schema",
            "max_output_tokens": 2000,
        }
    )
    message = asyncio.run(
        model._query_async(
            [{"role": "user", "content": "List the files in /tmp. Emit the next shell command."}]
        )
    )
    assert message["extra"]["actions"], f"{deployment} returned no action"
    assert message["extra"]["usage"]["last_response"]["output_tokens"] > 0
