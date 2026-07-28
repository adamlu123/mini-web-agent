from __future__ import annotations

from pathlib import Path
from typing import Any

from echo_rl.web_agent.prompts import (
    SFT_STATE_DEBUG_INSTRUCTIONS,
    SFT_STATE_DEBUG_SYSTEM,
)
from miniswewebagent.agents.default import DefaultAgent
from miniswewebagent.run.mini import _apply_prompt_mode


class _RecordingModel:
    def get_template_vars(self) -> dict[str, Any]:
        return {}

    def format_message(
        self,
        *,
        role: str,
        content: Any = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"role": role, "content": content, "extra": extra or {}}


class _TemplateEnvironment:
    def get_template_vars(self) -> dict[str, Any]:
        return {
            "workspace_dir": "/workspace",
            "final_script_path": "/workspace/final_script.py",
        }


def _capture_initial_messages(
    agent_config: dict[str, Any],
    *,
    task: str,
    task_id: str,
    start_url: str,
) -> list[dict[str, Any]]:
    agent = DefaultAgent(_RecordingModel(), _TemplateEnvironment(), **agent_config)
    captured: list[dict[str, Any]] = []

    def stop_after_initialization() -> None:
        captured.extend(agent.messages)
        agent.add_messages({"role": "exit", "content": "", "extra": {}})

    agent.step = stop_after_initialization  # type: ignore[method-assign]
    agent.save = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]
    agent.run(task=task, task_id=task_id, start_url=start_url)
    return captured


def test_sft_state_debug_default_agent_preserves_literal_training_system_prompt() -> None:
    config: dict[str, Any] = {"agent": {"prompt_mode": "sft_state_debug"}}
    _apply_prompt_mode(config)

    messages = _capture_initial_messages(
        config["agent"],
        task="Find the blue widget",
        task_id="example-task-17",
        start_url="https://example.test/start",
    )

    assert config["agent"]["render_system_template"] is False
    assert messages[0] == {"role": "system", "content": SFT_STATE_DEBUG_SYSTEM, "extra": {}}
    assert "{{ start_url }}" in messages[0]["content"]
    assert "{{ workspace_dir }}" in messages[0]["content"]


def test_sft_state_debug_default_agent_still_renders_instance_task_fields() -> None:
    config: dict[str, Any] = {"agent": {"prompt_mode": "sft_state_debug"}}
    _apply_prompt_mode(config)

    messages = _capture_initial_messages(
        config["agent"],
        task="Find the blue widget",
        task_id="example-task-17",
        start_url="https://example.test/start",
    )

    expected_user = (
        "Task: Find the blue widget\n"
        "Task ID: example-task-17\n"
        "Start URL: https://example.test/start\n"
        "Workspace root: /workspace\n"
        "Task metadata JSON: /workspace/task.json\n"
        "Required final script path: /workspace/final_script.py\n\n"
        + SFT_STATE_DEBUG_INSTRUCTIONS
    )
    assert messages[1] == {"role": "user", "content": expected_user, "extra": {}}


def test_default_agent_renders_system_template_by_default() -> None:
    messages = _capture_initial_messages(
        {
            "system_template": "Task={{ task }}; workspace={{ workspace_dir }}",
            "instance_template": "Open {{ start_url }}",
        },
        task="render me",
        task_id="unused",
        start_url="https://example.test/default",
    )

    assert messages[0]["content"] == "Task=render me; workspace=/workspace"
    assert messages[1]["content"] == "Open https://example.test/default"
