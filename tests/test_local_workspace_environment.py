from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from miniswewebagent.environments.local_workspace import LocalWorkspaceEnvironment
from miniswewebagent.run.mini import run_one


def test_local_workspace_environment_executes_single_command_and_summarizes_artifacts(tmp_path: Path) -> None:
    cred_path = tmp_path / "cred.sh"
    cred_path.write_text("export TEST_SECRET=workspace-secret\n", encoding="utf-8")

    env = LocalWorkspaceEnvironment(
        output_dir=tmp_path / "workspace",
        credentials_file=cred_path,
        env={"EXTRA_FLAG": "1"},
        command_timeout_seconds=10,
    )
    env.prepare(task="probe", task_id="probe", start_url="https://example.com")

    tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0pL1sAAAAASUVORK5CYII="
    command = f"""python - <<'PY'
from pathlib import Path
import base64
import os

workspace = Path(os.environ["WORKSPACE_DIR"])
(workspace / "screenshots").mkdir(parents=True, exist_ok=True)
(workspace / "final_script.py").write_text("print('workspace run ok')\\n", encoding="utf-8")
(workspace / "screenshots" / "after.png").write_bytes(base64.b64decode("{tiny_png}"))
print("task_json_exists", Path(os.environ["OM2W_TASK_JSON"]).exists())
print("test_secret", os.environ["TEST_SECRET"])
print("extra_flag", os.environ["EXTRA_FLAG"])
PY"""

    result = env.execute({"python_code": command})

    assert result["returncode"] == 0
    assert "workspace-secret" in result["output"]
    assert result["observation"]["final_script_exists"] is True
    assert result["observation"]["final_script_preview"] == "print('workspace run ok')\n"
    assert result["observation"]["screenshot_path"].endswith("after.png")
    assert "screenshots/after.png" in result["observation"]["recent_screenshots"]
    assert "final_script.py" in result["observation"]["workspace_files"]
    assert (tmp_path / "workspace" / "steps" / "step_0001.sh").exists()
    assert (tmp_path / "workspace" / "logs" / "step_0001.log").read_text(encoding="utf-8") == result["output"]

    task_payload = json.loads((tmp_path / "workspace" / "task.json").read_text(encoding="utf-8"))
    assert task_payload["task_id"] == "probe"
    env.close()


class _FakeWorkspaceGatewayHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        type(self).calls += 1
        if type(self).calls == 1:
            content = """
<response>
  <thought>Write and run the first draft of final_script.py.</thought>
  <python_code><![CDATA[
cat > final_script.py <<'PY'
print("workspace run ok")
PY
python final_script.py
  ]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
            """.strip()
        else:
            content = """
<response>
  <thought>The task is complete.</thought>
  <python_code><![CDATA[
  ]]></python_code>
  <done>true</done>
  <final_response>completed via workspace harness</final_response>
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


def test_run_one_supports_workspace_harness_config(tmp_path: Path) -> None:
    _FakeWorkspaceGatewayHandler.calls = 0
    server = HTTPServer(("127.0.0.1", 0), _FakeWorkspaceGatewayHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        result = run_one(
            task="Create a reproducible final script artifact.",
            start_url="https://example.com",
            output_dir=tmp_path / "artifacts",
            config_spec=[
                "benchmark/om2w_hard_local_workspace.yaml",
                "model.openai_gateway_api_key=dummy",
                f"model.openai_gateway_endpoint=http://127.0.0.1:{server.server_port}",
                "run.judge_enabled=false",
            ],
        )
    finally:
        server.shutdown()
        server.server_close()

    output_dir = Path(result["_output_dir"])
    assert result["final_response"] == "completed via workspace harness"
    assert (output_dir / "final_script.py").read_text(encoding="utf-8") == 'print("workspace run ok")\n'
    assert (output_dir / "steps" / "step_0001.sh").exists()
    assert "workspace run ok" in (output_dir / "logs" / "step_0001.log").read_text(encoding="utf-8")
