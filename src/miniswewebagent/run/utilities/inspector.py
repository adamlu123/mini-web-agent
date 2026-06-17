#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import typer
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _messages_to_steps(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    steps: list[list[dict[str, Any]]] = []
    current_step: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and current_step:
            steps.append(current_step)
            current_step = [message]
        else:
            current_step.append(message)
    if current_step:
        steps.append(current_step)
    return steps


def _render_message(message: dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    extra = message.get("extra", {})
    content = message.get("content", "")
    parts = [f"role: {role}"]

    if role == "assistant":
        parts.append("thought:")
        parts.append(str(content))
        actions = extra.get("actions", [])
        if actions:
            parts.append("")
            parts.append("generated code:")
            parts.append(
                actions[0].get("bash_command", "") or actions[0].get("command", "") or actions[0].get("python_code", "")
            )
        if extra.get("done"):
            parts.append("")
            parts.append(f"done: {extra.get('done')}")
            parts.append(f"final_response: {extra.get('final_response', '')}")
    elif role == "user" and isinstance(extra.get("observation"), dict):
        observation = extra["observation"]
        parts.extend(
            [
                f"status: {'ok' if observation.get('success') else 'error'}",
                f"url: {observation.get('url', '')}",
                f"title: {observation.get('title', '')}",
                f"screenshot_path: {observation.get('screenshot_path', '')}",
                "",
                "console output:",
                observation.get("console_output", "") or "<none>",
                "",
                "aria snapshot:",
                observation.get("aria_snapshot", "") or "<none>",
            ]
        )
    else:
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            parts.append("\n".join(text_parts))
        else:
            parts.append(str(content))
    return "\n".join(parts)


class BindingCommandProvider(Provider):
    COMMAND_DESCRIPTIONS = {
        "next_step": "Next step in the current trajectory",
        "previous_step": "Previous step in the current trajectory",
        "first_step": "First step in the current trajectory",
        "last_step": "Last step in the current trajectory",
        "scroll_down": "Scroll down",
        "scroll_up": "Scroll up",
        "next_trajectory": "Next trajectory",
        "previous_trajectory": "Previous trajectory",
        "open_in_jless": "Open the current step in jless",
        "open_in_jless_all": "Open the entire trajectory in jless",
        "quit": "Quit the inspector",
    }

    async def discover(self) -> Hits:
        for binding in self.app.BINDINGS:
            desc = self.COMMAND_DESCRIPTIONS.get(binding.action, binding.description)
            yield DiscoveryHit(desc, lambda b=binding: self.app.run_action(b.action))

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for binding in self.app.BINDINGS:
            desc = self.COMMAND_DESCRIPTIONS.get(binding.action, binding.description)
            score = matcher.match(desc)
            if score > 0:
                yield Hit(score, matcher.highlight(desc), lambda b=binding: self.app.run_action(b.action))


class TrajectoryInspector(App):
    COMMANDS = {BindingCommandProvider}
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "↓"),
        Binding("k,up", "scroll_up", "↑"),
        Binding("L", "next_trajectory", "Traj++"),
        Binding("H", "previous_trajectory", "Traj--"),
        Binding("e", "open_in_jless", "Jless"),
        Binding("E", "open_in_jless_all", "Jless (all)"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, trajectory_files: list[Path]):
        css_path = os.environ.get(
            "MSWEBA_INSPECTOR_STYLE_PATH",
            str(Path(__file__).parents[2] / "config" / "inspector.tcss"),
        )
        self.__class__.CSS = Path(css_path).read_text()
        super().__init__()
        self.trajectory_files = trajectory_files
        self._i_trajectory = 0
        self._i_step = 0
        self.messages: list[dict[str, Any]] = []
        self.steps: list[list[dict[str, Any]]] = []
        if trajectory_files:
            self._load_current_trajectory()

    @property
    def i_step(self) -> int:
        return self._i_step

    @i_step.setter
    def i_step(self, value: int) -> None:
        if value != self._i_step and self.n_steps > 0:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def i_trajectory(self) -> int:
        return self._i_trajectory

    @i_trajectory.setter
    def i_trajectory(self, value: int) -> None:
        if value != self._i_trajectory and self.n_trajectories > 0:
            self._i_trajectory = max(0, min(value, self.n_trajectories - 1))
            self._load_current_trajectory()
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectory_files)

    def _load_current_trajectory(self) -> None:
        if not self.trajectory_files:
            self.messages = []
            self.steps = []
            return
        trajectory_file = self.trajectory_files[self.i_trajectory]
        try:
            data = json.loads(trajectory_file.read_text())
            if isinstance(data, dict) and "messages" in data:
                self.messages = data["messages"]
            elif isinstance(data, list):
                self.messages = data
            else:
                raise ValueError("Unrecognized trajectory format")
            self.steps = _messages_to_steps(self.messages)
            self._i_step = 0
        except Exception as exc:
            self.messages = []
            self.steps = []
            self.notify(f"Error loading {trajectory_file.name}: {exc}", severity="error")

    @property
    def current_trajectory_name(self) -> str:
        if not self.trajectory_files:
            return "No trajectories"
        return self.trajectory_files[self.i_trajectory].name

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.update_content()

    def update_content(self) -> None:
        container = self.query_one("#content", Vertical)
        container.remove_children()
        if not self.steps:
            container.mount(Static("No trajectory loaded or empty trajectory"))
            self.title = "Trajectory Inspector - No Data"
            return
        for message in self.steps[self.i_step]:
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message.get("role") or "unknown"
            message_container.mount(Static(role.upper(), classes="message-header"))
            message_container.mount(
                Static(Text.from_ansi(_render_message(message).replace("\x00", ""), no_wrap=False), classes="message-content")
            )
        self.title = (
            f"Trajectory {self.i_trajectory + 1}/{self.n_trajectories} - "
            f"{self.current_trajectory_name} - Step {self.i_step + 1}/{self.n_steps}"
        )

    def action_next_step(self) -> None:
        self.i_step += 1

    def action_previous_step(self) -> None:
        self.i_step -= 1

    def action_first_step(self) -> None:
        self.i_step = 0

    def action_last_step(self) -> None:
        self.i_step = self.n_steps - 1

    def action_next_trajectory(self) -> None:
        self.i_trajectory += 1

    def action_previous_trajectory(self) -> None:
        self.i_trajectory -= 1

    def action_scroll_down(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)

    def _open_in_jless(self, path: Path) -> None:
        with self.suspend():
            try:
                subprocess.run(["jless", path])
            except FileNotFoundError:
                self.notify("jless not found. Install with: brew install jless", severity="error")

    def action_open_in_jless(self) -> None:
        if not self.steps:
            self.notify("No messages to display", severity="warning")
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(self.steps[self.i_step], handle, indent=2)
            temp_path = Path(handle.name)
        self._open_in_jless(temp_path)
        temp_path.unlink()

    def action_open_in_jless_all(self) -> None:
        if not self.trajectory_files:
            self.notify("No trajectory to display", severity="warning")
            return
        self._open_in_jless(self.trajectory_files[self.i_trajectory])


@app.command()
def main(path: str = typer.Argument(".", help="Directory to search for trajectories or a specific trajectory file")) -> None:
    path_obj = Path(path)
    if path_obj.is_file():
        trajectory_files = [path_obj]
    elif path_obj.is_dir():
        trajectory_files = sorted(path_obj.rglob("trajectory.json")) + sorted(path_obj.rglob("*.traj.json"))
        if not trajectory_files:
            raise typer.BadParameter(f"No trajectory files found in '{path}'")
    else:
        raise typer.BadParameter(f"Path '{path}' does not exist")
    TrajectoryInspector(trajectory_files).run()


if __name__ == "__main__":
    app()
