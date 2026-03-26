from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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
    steps_md = (tmp_path / "artifacts" / next((tmp_path / "artifacts").iterdir()).name / "debug" / "steps.md")
    steps_md_text = steps_md.read_text(encoding="utf-8")
    assert '"console_output"' in steps_md_text
    assert '"recent_console"' in steps_md_text
    assert '"aria_snapshot"' not in steps_md_text


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
