from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from miniswewebagent import Environment, Model, __version__
from miniswewebagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded
from miniswewebagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 15
    debug_log: bool = True
    output_path: Path | None = None


def _sanitize_message_for_disk(message: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(message)
    content = cloned.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                part["image_url"] = "<omitted:data-url>"
    return cloned


def _observation_for_markdown(observation: dict[str, Any], *, model_usage: dict[str, Any] | None = None) -> dict[str, Any]:
    cloned = copy.deepcopy(observation)
    cloned.pop("aria_snapshot", None)
    if model_usage:
        cloned["model_usage"] = copy.deepcopy(model_usage)
    return cloned


def _action_text(action: dict[str, Any]) -> str:
    return str(action.get("bash_command") or action.get("command") or action.get("python_code") or "").strip()


def _python_action_text(action: dict[str, Any]) -> str:
    return str(action.get("python_code") or "").strip()


def _markdown_code_fence_language(*, bash_command_text: str, python_code_text: str) -> str:
    if bash_command_text:
        return "bash"
    if python_code_text:
        return "python"
    return ""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict[str, Any]] = []
        self.model = model
        self.env = env
        self.extra_template_vars: dict[str, Any] = {}
        self.n_calls = 0
        self.n_format_errors = 0

    def _debug_dir(self) -> Path | None:
        if self.config.output_path is None:
            return None
        return self.config.output_path.parent / "debug"

    def _write_debug_step_artifact(
        self,
        *,
        step_index: int,
        assistant_message: dict[str, Any],
        outputs: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.config.debug_log:
            return
        debug_dir = self._debug_dir()
        if debug_dir is None:
            return
        steps_dir = debug_dir / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        extra = assistant_message.get("extra", {})
        actions = extra.get("actions", [])
        action_text = "\n\n".join(_action_text(action) for action in actions if _action_text(action))
        python_code_text = "\n\n".join(
            _python_action_text(action) for action in actions if _python_action_text(action)
        )
        bash_command_text = "\n\n".join(
            str(action.get("bash_command", "")).strip()
            for action in actions
            if str(action.get("bash_command", "")).strip()
        )
        code_fence_language = _markdown_code_fence_language(
            bash_command_text=bash_command_text,
            python_code_text=python_code_text,
        )
        payload = {
            "step": step_index,
            "thought": assistant_message.get("content", ""),
            "python_code": python_code_text,
            "bash_command": bash_command_text,
            "command_text": action_text,
            "raw_response": extra.get("raw_response", {}),
            "done": extra.get("done", False),
            "final_response": extra.get("final_response", ""),
            "outputs": outputs or [],
        }
        (steps_dir / f"step_{step_index:04d}.json").write_text(json.dumps(payload, indent=2))

        summary_path = debug_dir / "steps.md"
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(f"## Step {step_index}\n\n")
            handle.write("### Thought\n\n")
            handle.write(f"{payload['thought']}\n\n")
            handle.write("### Generated Code\n\n")
            handle.write(f"```{code_fence_language}\n")
            handle.write(f"{payload['command_text']}\n")
            handle.write("```\n\n")
            if outputs:
                observation = outputs[0].get("observation", {})
                markdown_observation = _observation_for_markdown(
                    observation,
                    model_usage=extra.get("usage"),
                )
                handle.write("### Observation\n\n")
                handle.write("```json\n")
                handle.write(f"{json.dumps(markdown_observation, indent=2)}\n")
                handle.write("```\n\n")

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict[str, Any]) -> list[dict[str, Any]]:
        self.messages.extend(messages)
        return list(messages)

    def run(self, task: str = "", **kwargs) -> dict[str, Any]:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.n_calls = 0
        self.n_format_errors = 0
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
            except InterruptAgentFlow as exc:
                if isinstance(exc, FormatError):
                    self.n_format_errors += 1
                self.add_messages(*exc.messages)
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict[str, Any]]:
        return self.execute_actions(self.query())

    def query(self) -> dict[str, Any]:
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                self.model.format_message(
                    role="exit",
                    content="Step limit exceeded.",
                    extra={"exit_status": "LimitsExceeded", "submission": ""},
                )
            )
        message = self.model.query(self.messages)
        self.n_calls += 1
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        extra = message.get("extra", {})
        if extra.get("done"):
            self._write_debug_step_artifact(step_index=self.n_calls, assistant_message=message, outputs=[])
            return self.add_messages(
                self.model.format_message(
                    role="exit",
                    content=extra.get("final_response", "Task completed."),
                    extra={
                        "exit_status": "Submitted",
                        "submission": extra.get("final_response", ""),
                        "final_response": extra.get("final_response", ""),
                    },
                )
            )
        outputs = [self.env.execute(action) for action in extra.get("actions", [])]
        self._write_debug_step_artifact(step_index=self.n_calls, assistant_message=message, outputs=outputs)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict[str, Any]:
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        return recursive_merge(
            {
                "info": {
                    "config": {
                        "agent": self.config.model_dump(mode="json"),
                        "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    },
                    "mini_version": __version__,
                    "exit_status": last_extra.get("exit_status", ""),
                    "submission": last_extra.get("submission", ""),
                    "api_calls": self.n_calls,
                    "format_errors": self.n_format_errors,
                },
                "messages": [_sanitize_message_for_disk(message) for message in self.messages],
                "trajectory_format": "mini-swe-webagent-0.1",
            },
            self.model.serialize(),
            self.env.serialize(),
            *extra_dicts,
        )

    def save(self, path: Path | None, *extra_dicts) -> dict[str, Any]:
        data = self.serialize(*extra_dicts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
