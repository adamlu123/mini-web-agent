"""Tests for the max_context_tokens eviction path and judge discovery robustness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from miniswewebagent.agents.default import DefaultAgent
from miniswewebagent.models.openrouter_model import (
    MIN_CHARS_PER_TOKEN,
    TEMPLATE_TOKENS_PER_MESSAGE,
    OpenRouterModel,
    OpenRouterModelConfig,
    _serialized_text_chars,
    _serialized_utf8_bytes,
    _tokenize_endpoints,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "eval_persistent_cli_with_original_om2w.py"
)


def _load_judge_module():
    spec = importlib.util.spec_from_file_location("_judge_mod", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, whose field
    # resolution looks the defining module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubModel:
    """Counts tokens as one per four characters of serialized content."""

    def __init__(self) -> None:
        self.calls = 0

    def format_message(self, *, role, content="", extra=None):
        return {"role": role, "content": content, "extra": extra or {}}

    def get_template_vars(self):
        return {}

    def prompt_tokens_within(self, messages, budget):
        self.calls += 1
        chars = sum(len(m.get("content") or "") for m in messages)
        return chars // 4 <= budget


class _StubEnvironment:
    def get_template_vars(self, **kwargs):
        return {"workspace_dir": "/workspace", **kwargs}


def _make_agent(**kwargs) -> DefaultAgent:
    return DefaultAgent(
        _StubModel(),
        _StubEnvironment(),
        system_template="system",
        instance_template="instance",
        **kwargs,
    )


def _history(n_turns: int, chars: int = 400) -> list[dict]:
    messages = [
        {"role": "system", "content": "S" * 40},
        {"role": "user", "content": "TASK"},
    ]
    for i in range(n_turns):
        messages.append({"role": "assistant", "content": f"a{i}" + "A" * chars, "extra": {}})
        messages.append({"role": "user", "content": f"Command output:\n{'O' * chars}"})
    return messages


def test_no_budget_leaves_history_untouched() -> None:
    agent = _make_agent(max_context_tokens=0)
    messages = _history(30)
    assert agent._fit_context(messages) == agent._transform_history(
        agent._windowed_messages(messages)
    )
    assert agent.model.calls == 0


def test_request_under_budget_is_not_evicted() -> None:
    agent = _make_agent(max_context_tokens=1_000_000)
    messages = _history(30)
    assert len(agent._fit_context(messages)) == len(messages)


def test_over_budget_history_is_evicted_to_fit() -> None:
    budget = 2000
    agent = _make_agent(max_context_tokens=budget)
    messages = _history(60)
    fitted = agent._fit_context(messages)

    assert len(fitted) < len(messages)
    assert sum(len(m.get("content") or "") for m in fitted) // 4 <= budget
    # System prompt and the task are always preserved.
    assert fitted[0]["role"] == "system"
    assert "TASK" in fitted[1]["content"]
    # The most recent turn must survive eviction.
    assert fitted[-1]["content"] == messages[-1]["content"]


def test_eviction_keeps_turns_alternating() -> None:
    agent = _make_agent(max_context_tokens=2000)
    fitted = agent._fit_context(_history(60))
    roles = [m["role"] for m in fitted]
    assert roles[0] == "system"
    assert all(a != b for a, b in zip(roles[1:], roles[2:])), roles


def test_eviction_is_non_mutating() -> None:
    agent = _make_agent(max_context_tokens=2000)
    messages = _history(60)
    before = json.dumps(messages)
    agent._fit_context(messages)
    assert json.dumps(messages) == before


def test_eviction_respects_configured_window_as_upper_bound() -> None:
    agent = _make_agent(max_context_tokens=2000, context_window_steps=4)
    fitted = agent._fit_context(_history(60))
    assert sum(1 for m in fitted if m["role"] == "assistant") <= 4


def test_model_without_counter_is_left_alone() -> None:
    class _NoCounterModel(_StubModel):
        prompt_tokens_within = None

    agent = _make_agent(max_context_tokens=1)
    agent.model = _NoCounterModel()
    messages = _history(20)
    assert len(agent._fit_context(messages)) == len(messages)


def test_char_bound_never_underestimates_tokens() -> None:
    # The pre-filter is only safe if chars / MIN_CHARS_PER_TOKEN >= real tokens.
    assert MIN_CHARS_PER_TOKEN <= 2.36
    serialized = [{"role": "user", "content": "x" * 100}]
    assert _serialized_text_chars(serialized) == 100


def test_utf8_byte_bound_counts_multibyte_text() -> None:
    serialized = [{"role": "user", "content": "\u4f60\u597d"}]
    assert _serialized_text_chars(serialized) == 2
    assert _serialized_utf8_bytes(serialized) == 6


def test_utf8_byte_bound_counts_parts() -> None:
    serialized = [{"role": "user", "content": [{"type": "text", "text": "abc"}]}]
    assert _serialized_utf8_bytes(serialized) == 3


def test_dense_payload_is_not_short_circuited() -> None:
    """Regression: a 2.36 chars/token payload slipped past a chars/2.5 filter.

    69175 characters tokenized to 29289 with a 28000 budget, but the character
    estimate said 27670, so the exact count was never requested and the request
    was sent over-length.
    """

    class _Model(OpenRouterModel):
        def __init__(self) -> None:
            self.config = OpenRouterModelConfig(
                openrouter_api_key="k", response_mode="sft_state"
            )
            self._tokenize_url = None
            self.exact_calls = 0

        def _exact_prompt_tokens(self, serialized):
            self.exact_calls += 1
            return 29289

    model = _Model()
    messages = [{"role": "user", "content": "x" * 69175}]
    assert model.prompt_tokens_within(messages, 28000) is False
    assert model.exact_calls == 1, "the exact count must not be short-circuited"


def test_small_request_skips_the_exact_count() -> None:
    class _Model(OpenRouterModel):
        def __init__(self) -> None:
            self.config = OpenRouterModelConfig(
                openrouter_api_key="k", response_mode="sft_state"
            )
            self._tokenize_url = None
            self.exact_calls = 0

        def _exact_prompt_tokens(self, serialized):
            self.exact_calls += 1
            return 10

    model = _Model()
    messages = [{"role": "user", "content": "x" * 100}]
    assert model.prompt_tokens_within(messages, 28000) is True
    assert model.exact_calls == 0
    assert TEMPLATE_TOKENS_PER_MESSAGE > 0


def test_tokenize_endpoint_candidates() -> None:
    # vLLM serves /tokenize at the root; the /v1 variant 404s, so root goes first.
    assert _tokenize_endpoints("http://127.0.0.1:8000/v1/chat/completions") == [
        "http://127.0.0.1:8000/tokenize",
        "http://127.0.0.1:8000/v1/tokenize",
    ]
    assert _tokenize_endpoints("http://host:9000/chat/completions") == [
        "http://host:9000/tokenize"
    ]


def _write_task_dir(root: Path, task_id: str) -> Path:
    task_dir = root / task_id
    (task_dir / "screenshots").mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": task_id, "task": "do a thing"}), encoding="utf-8"
    )
    return task_dir


def test_discovery_skips_agent_created_symlink(tmp_path: Path) -> None:
    module = _load_judge_module()
    task_dir = _write_task_dir(tmp_path, "abc123")
    # Reproduces `ln -sfn /workspace /workspace_backup` run inside the task dir.
    (tmp_path / "abc123_backup").symlink_to(task_dir, target_is_directory=True)

    artifacts = module.discover_task_artifacts(tmp_path)
    assert [a.task_id for a in artifacts] == ["abc123"]


def test_discovery_keeps_first_on_duplicate_task_id(tmp_path: Path, capsys) -> None:
    module = _load_judge_module()
    _write_task_dir(tmp_path, "abc123")
    other = tmp_path / "abc123_copy"
    (other / "screenshots").mkdir(parents=True)
    (other / "task.json").write_text(
        json.dumps({"task_id": "abc123", "task": "do a thing"}), encoding="utf-8"
    )

    artifacts = module.discover_task_artifacts(tmp_path)
    assert [a.task_id for a in artifacts] == ["abc123"]
    assert artifacts[0].task_dir == str((tmp_path / "abc123").resolve())
    assert "duplicate task ID" in capsys.readouterr().out
