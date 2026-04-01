from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miniswewebagent.run.utilities.trace_viewer import TraceCatalog


def test_trace_catalog_lists_runs_tasks_and_detail(tmp_path) -> None:
    root = tmp_path / "outputs" / "default"
    task_dir = root / "run_a" / "task_001"
    (task_dir / "debug" / "steps").mkdir(parents=True)
    (task_dir / "screenshots").mkdir(parents=True)
    (task_dir / "trajectory").mkdir(parents=True)

    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-001",
                "task": "Search for a thing.",
                "start_url": "https://example.com",
                "final_result_response": "Found it.",
                "exit_status": "Submitted",
                "action_history": ["await page.goto('https://example.com')"],
                "thoughts": ["Open the page."],
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "debug" / "steps" / "step_0001.json").write_text(
        json.dumps(
            {
                "step": 1,
                "thought": "Open the page.",
                "python_code": "await page.goto('https://example.com')",
                "done": False,
                "outputs": [
                    {
                        "observation": {
                            "success": True,
                            "url": "https://example.com",
                            "title": "Example Domain",
                            "screenshot_path": "outputs/default/run_a/task_001/screenshots/step_0001.png",
                            "console_output": "console line",
                            "aria_snapshot": "- document",
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "screenshots" / "step_0001.png").write_bytes(b"png")
    (task_dir / "trajectory" / "0_full_screenshot.png").write_bytes(b"png")
    (root / "run_a" / "WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json").write_text(
        json.dumps(
            {
                "task_id": "task-001",
                "predicted_label": 1,
                "evaluation_details": {
                    "response": "Thoughts: The task was completed.\nStatus: success",
                    "predicted_label": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    catalog = TraceCatalog(root)

    runs = catalog.list_runs()
    assert [run["id"] for run in runs] == ["run_a"]

    tasks = catalog.list_tasks("run_a")
    assert len(tasks) == 1
    assert tasks[0]["taskId"] == "task-001"
    assert tasks[0]["status"] == "submitted"

    detail = catalog.task_detail("run_a", "task_001")
    assert detail["taskId"] == "task-001"
    assert detail["status"] == "submitted"
    assert detail["stepCount"] == 1
    assert detail["steps"][0]["url"] == "https://example.com"
    assert detail["steps"][0]["screenshotRelPath"] == "screenshots/step_0001.png"
    assert len(detail["judges"]) == 1
    assert detail["judges"][0]["model"] == "o4-mini"
    assert detail["judges"][0]["status"] == "success"
    assert detail["judges"][0]["response"] == "Thoughts: The task was completed.\nStatus: success"
