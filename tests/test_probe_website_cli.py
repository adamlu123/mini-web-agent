from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from typer.testing import CliRunner

from miniswewebagent.run.utilities.probe_website import app


class _FakeProbeGatewayHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        type(self).calls += 1

        if type(self).calls == 1:
            content = """
<response>
  <thought>Write the probe script and selector summary.</thought>
  <bash_command><![CDATA[
cat > probe_script.py <<'PY'
print("probe ok")
PY
cat > probe_summary.json <<'JSON'
{
  "summary": "Dedicated year and sort controls are visible on the results surface.",
  "site_status": "ok",
  "final_url": "https://example.com/results",
  "controls": [
    {
      "constraint": "year",
      "state": "found",
      "surface": "results page",
      "control_label": "Year",
      "recommended_locator": "page.get_by_role('button', name='Year')",
      "recommended_action": "Open the Year menu and choose the exact year value.",
      "evidence": "Visible Year filter button in the results toolbar."
    },
    {
      "constraint": "sort",
      "state": "found",
      "surface": "results page",
      "control_label": "Sort by",
      "recommended_locator": "page.get_by_role('combobox', name='Sort by')",
      "recommended_action": "Select the requested ranking option from the sort combobox.",
      "evidence": "Visible Sort by combobox near the result count."
    }
  ],
  "blockers": [],
  "suggested_next_actions": [
    "Navigate directly to the results surface before editing final_script.py.",
    "Apply year through the Year control instead of URL parameters."
  ],
  "artifacts": {
    "probe_script": "probe_script.py",
    "screenshots": [],
    "logs": []
  }
}
JSON
cat > probe_summary.md <<'MD'
# Probe

- Year control is visible on results page.
- Sort by combobox is visible on results page.
MD
  ]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
            """.strip()
        else:
            content = """
<response>
  <thought>The probe report is complete.</thought>
  <bash_command><![CDATA[
  ]]></bash_command>
  <done>true</done>
  <final_response>probe finished; probe_summary.json is the final artifact</final_response>
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


def test_probe_website_cli_writes_parent_reports(tmp_path: Path) -> None:
    _FakeProbeGatewayHandler.calls = 0
    server = HTTPServer(("127.0.0.1", 0), _FakeProbeGatewayHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    parent_workspace = tmp_path / "workspace"
    parent_workspace.mkdir(parents=True, exist_ok=True)
    (parent_workspace / "task.json").write_text(
        json.dumps(
            {
                "task": "Find the correct used-car filters.",
                "task_id": "cars-task",
                "start_url": "https://example.com/cars",
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    try:
        result = runner.invoke(
            app,
            [
                "--workspace-dir",
                str(parent_workspace),
                "--objective",
                "Inspect whether the site exposes dedicated year and sort controls.",
                "--max-steps",
                "4",
                "-c",
                "model.openai_gateway_api_key=dummy",
                "-c",
                f"model.openai_gateway_endpoint=http://127.0.0.1:{server.server_port}",
                "-c",
                "environment.credentials_file=null",
            ],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0

    latest_json = json.loads((parent_workspace / "probe_reports" / "latest.json").read_text(encoding="utf-8"))
    latest_markdown = (parent_workspace / "probe_reports" / "latest.md").read_text(encoding="utf-8")

    assert latest_json["site_status"] == "ok"
    assert latest_json["controls"][0]["constraint"] == "year"
    assert latest_json["report_paths"]["latest_markdown_path"].endswith("probe_reports/latest.md")
    assert "Sort by combobox is visible on results page." in latest_markdown
    assert "probe_runs" in latest_json["probe_run_dir"]
    assert "probe_reports" in result.stdout
