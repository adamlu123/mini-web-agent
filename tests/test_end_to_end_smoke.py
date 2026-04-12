from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from miniswewebagent.run.mini import run_one


class _FakeGatewayHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        messages = payload.get("input", [])
        type(self).calls += 1

        if type(self).calls == 1:
            content = """
<response>
  <thought>Capture the current page state.</thought>
  <python_code><![CDATA[
await page.wait_for_load_state('domcontentloaded')
]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
            """.strip()
        else:
            observation_text = ""
            last_content = messages[-1].get("content", [])
            if isinstance(last_content, list):
                for part in last_content:
                    if part.get("type") == "input_text":
                        observation_text += part.get("text", "") + "\n"

            title = ""
            heading = ""
            for line in observation_text.splitlines():
                if line.startswith("Title: "):
                    title = line.removeprefix("Title: ").strip()
                if 'heading "Example Domain"' in line:
                    heading = "Example Domain"

            content = f"""
<response>
  <thought>The task is complete.</thought>
  <python_code><![CDATA[
]]></python_code>
  <done>true</done>
  <final_response>Title: {title}; Heading: {heading}</final_response>
</response>
            """.strip()

        response = {
            "usage": {
                "input_tokens": 17,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 9,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 26,
            },
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            ]
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        pass


def test_run_one_completes_live_page_task_with_fake_gateway(tmp_path) -> None:
    _FakeGatewayHandler.calls = 0
    server = HTTPServer(("127.0.0.1", 0), _FakeGatewayHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        result = run_one(
            task="Tell me the page title and main heading.",
            start_url="https://example.com",
            output_dir=tmp_path / "artifacts",
            config_spec=[
                "mini.yaml",
                "model.openai_gateway_api_key=dummy",
                f"model.openai_gateway_endpoint=http://127.0.0.1:{server.server_port}",
                "environment.browserbase_enabled=false",
                "environment.headless=true",
                "environment.slow_mo_ms=0",
                "environment.browser_timeout_ms=5000",
                "environment.browser_navigation_timeout_ms=10000",
                "environment.observation_timeout_ms=3000",
            ],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result["final_response"] == "Title: Example Domain; Heading: Example Domain"
    assert (tmp_path / "artifacts").exists()
    run_dir = tmp_path / "artifacts" / next((tmp_path / "artifacts").iterdir()).name
    steps_md = run_dir / "debug" / "steps.md"
    steps_md_text = steps_md.read_text(encoding="utf-8")
    assert '"console_output"' in steps_md_text
    assert '"recent_console"' in steps_md_text
    assert '"model_usage"' in steps_md_text
    assert '"input_tokens": 17' in steps_md_text
    assert '"text_chars"' not in steps_md_text
    assert '"serialized_chars"' not in steps_md_text
    assert '"aria_snapshot"' not in steps_md_text
    assert (run_dir / "config_snapshot" / "config_spec_manifest.json").exists()
    assert (run_dir / "config_snapshot" / "merged_config.yaml").exists()


def test_default_agent_resets_step_counter_between_runs() -> None:
    from miniswewebagent.agents.default import DefaultAgent

    class DummyModel:
        def __init__(self) -> None:
            self.calls = 0

        def get_template_vars(self, **kwargs):
            return {}

        def format_message(self, **kwargs):
            return {
                "role": kwargs["role"],
                "content": kwargs.get("content", ""),
                "extra": kwargs.get("extra", {}),
            }

        def query(self, messages, **kwargs):
            self.calls += 1
            return self.format_message(
                role="assistant",
                content="done",
                extra={
                    "actions": [],
                    "done": True,
                    "final_response": f"run {self.calls}",
                    "raw_response": {},
                },
            )

        def format_observation_messages(self, message, outputs, template_vars=None):
            return []

        def serialize(self):
            return {}

    class DummyEnv:
        def get_template_vars(self, **kwargs):
            return {}

        def execute(self, action, cwd=""):
            return {}

        def serialize(self):
            return {}

    agent = DefaultAgent(DummyModel(), DummyEnv(), system_template="x", instance_template="y", step_limit=1)

    first = agent.run("task 1")
    second = agent.run("task 2")

    assert first["final_response"] == "run 1"
    assert second["final_response"] == "run 2"


def test_default_agent_does_not_count_format_errors_toward_step_limit() -> None:
    from miniswewebagent.agents.default import DefaultAgent
    from miniswewebagent.exceptions import FormatError

    class DummyModel:
        def __init__(self) -> None:
            self.calls = 0

        def get_template_vars(self, **kwargs):
            return {}

        def format_message(self, **kwargs):
            return {
                "role": kwargs["role"],
                "content": kwargs.get("content", ""),
                "extra": kwargs.get("extra", {}),
            }

        def query(self, messages, **kwargs):
            self.calls += 1
            if self.calls <= 3:
                raise FormatError(
                    self.format_message(
                        role="user",
                        content="bad format",
                        extra={"interrupt_type": "FormatError", "model_response": "plain text"},
                    )
                )
            return self.format_message(
                role="assistant",
                content="done",
                extra={
                    "actions": [],
                    "done": True,
                    "final_response": "ok",
                    "raw_response": {},
                },
            )

        def format_observation_messages(self, message, outputs, template_vars=None):
            return []

        def serialize(self):
            return {}

    class DummyEnv:
        def get_template_vars(self, **kwargs):
            return {}

        def execute(self, action, cwd=""):
            return {}

        def serialize(self):
            return {}

    agent = DefaultAgent(DummyModel(), DummyEnv(), system_template="x", instance_template="y", step_limit=1)
    result = agent.run("task")

    assert result["final_response"] == "ok"
    assert agent.n_calls == 1
    assert agent.n_format_errors == 3


def test_debug_steps_markdown_uses_bash_fence_for_bash_actions(tmp_path) -> None:
    from miniswewebagent.agents.default import DefaultAgent

    class DummyModel:
        def get_template_vars(self, **kwargs):
            return {}

        def format_message(self, **kwargs):
            return {
                "role": kwargs["role"],
                "content": kwargs.get("content", ""),
                "extra": kwargs.get("extra", {}),
            }

        def query(self, messages, **kwargs):
            raise AssertionError("query should not be called in this test")

        def format_observation_messages(self, message, outputs, template_vars=None):
            return []

        def serialize(self):
            return {}

    class DummyEnv:
        def get_template_vars(self, **kwargs):
            return {}

        def execute(self, action, cwd=""):
            return {}

        def serialize(self):
            return {}

    output_path = tmp_path / "run" / "result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent = DefaultAgent(
        DummyModel(),
        DummyEnv(),
        system_template="x",
        instance_template="y",
        output_path=output_path,
    )

    assistant_message = {
        "role": "assistant",
        "content": "inspect files",
        "extra": {
            "actions": [{"bash_command": "ls -la", "command": "ls -la"}],
            "done": False,
            "final_response": "",
            "raw_response": {},
        },
    }

    agent._write_debug_step_artifact(step_index=1, assistant_message=assistant_message, outputs=[])

    steps_md = output_path.parent / "debug" / "steps.md"
    steps_md_text = steps_md.read_text(encoding="utf-8")
    assert "```bash\nls -la\n```" in steps_md_text


def test_run_one_closes_environment_when_prepare_fails(tmp_path, monkeypatch) -> None:
    events: list[str] = []

    class DummyAgent:
        messages: list[dict[str, object]] = []

        def run(self, *args, **kwargs):
            raise AssertionError("agent.run should not be called when prepare fails")

    class DummyEnvironment:
        def prepare(self, **kwargs) -> None:
            events.append("prepare")
            raise RuntimeError("prepare failed")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr("miniswewebagent.run.mini.get_model", lambda config: object())
    dummy_env = DummyEnvironment()
    monkeypatch.setattr("miniswewebagent.run.mini.get_environment", lambda config: dummy_env)
    monkeypatch.setattr(
        "miniswewebagent.run.mini.get_agent",
        lambda model, env, config, default_type="default": DummyAgent(),
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        run_one(
            task="Probe prepare failure cleanup.",
            start_url="https://example.com",
            output_dir=tmp_path / "artifacts",
            config_spec=["mini.yaml"],
        )

    assert events == ["prepare", "close"]
    output_root = tmp_path / "artifacts"
    task_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(task_dirs) == 1
    result_path = task_dirs[0] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["task"] == "Probe prepare failure cleanup."
    assert result["exit_status"] == "RuntimeError"
    assert result["run_exception"] == "prepare failed"


def test_run_one_accepts_explicit_task_id_without_tasks_file(tmp_path, monkeypatch) -> None:
    prepare_calls: list[dict[str, object]] = []

    class DummyAgent:
        messages: list[dict[str, object]] = []

        def run(self, *args, **kwargs):
            return {
                "exit_status": "Submitted",
                "submission": "",
                "final_response": "ok",
            }

    class DummyEnvironment:
        def prepare(self, **kwargs) -> None:
            prepare_calls.append(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr("miniswewebagent.run.mini.get_model", lambda config: object())
    monkeypatch.setattr("miniswewebagent.run.mini.get_environment", lambda config: DummyEnvironment())
    monkeypatch.setattr(
        "miniswewebagent.run.mini.get_agent",
        lambda model, env, config, default_type="default": DummyAgent(),
    )
    monkeypatch.setattr("miniswewebagent.run.mini.export_online_mind2web_artifacts", lambda **kwargs: {})

    result = run_one(
        task="Search for a flight.",
        task_id="flight__one",
        start_url="https://example.com/flight",
        output_dir=tmp_path / "artifacts",
        config_spec=["mini.yaml"],
    )

    assert result["final_response"] == "ok"
    assert prepare_calls == [
        {
            "task": "Search for a flight.",
            "task_id": "flight__one",
            "start_url": "https://example.com/flight",
            "task_record": None,
        }
    ]
