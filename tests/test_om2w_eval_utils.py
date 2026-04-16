from __future__ import annotations

import json
import subprocess
from pathlib import Path

from miniswewebagent.utils.om2w_eval import (
    export_online_mind2web_artifacts,
    export_online_mind2web_artifacts_all_final_execution,
    normalize_online_mind2web_judge_results,
)


def test_export_online_mind2web_artifacts_creates_result_and_trajectory(tmp_path) -> None:
    output_dir = tmp_path / "task"
    (output_dir / "debug" / "steps").mkdir(parents=True)
    (output_dir / "screenshots").mkdir(parents=True)

    (output_dir / "debug" / "steps" / "step_0001.json").write_text(
        json.dumps(
            {
                "step": 1,
                "thought": "Open the homepage.",
                "python_code": "await page.goto('https://example.com')",
                "done": False,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "debug" / "steps" / "step_0002.json").write_text(
        json.dumps(
            {
                "step": 2,
                "thought": "Task is complete.",
                "python_code": "",
                "done": True,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "screenshots" / "step_0001.png").write_bytes(b"png")

    artifacts = export_online_mind2web_artifacts(
        output_dir=output_dir,
        task="Example task",
        task_id="task-1",
        start_url="https://example.com",
        agent_result={"final_response": "Done", "exit_status": "Submitted", "submission": "Done"},
    )

    result_path = output_dir / "result.json"
    assert artifacts["result_json"] == str(result_path)
    assert result_path.exists()

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["task_id"] == "task-1"
    assert payload["task"] == "Example task"
    assert payload["action_history"] == ["await page.goto('https://example.com')"]
    assert payload["thoughts"] == ["Open the homepage."]
    assert payload["final_result_response"] == "Done"
    assert len(payload["screenshot_paths"]) == 2

    exported_image = output_dir / "trajectory" / "0_full_screenshot.png"
    placeholder_image = output_dir / "trajectory" / "1_full_screenshot.png"
    assert artifacts["trajectory_dir"] == str(output_dir / "trajectory")
    assert exported_image.exists()
    assert placeholder_image.exists()
    assert placeholder_image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_export_online_mind2web_artifacts_uses_workspace_observation_screenshots(tmp_path) -> None:
    output_dir = tmp_path / "task"
    (output_dir / "debug" / "steps").mkdir(parents=True)
    (output_dir / "screenshots").mkdir(parents=True)

    screenshot_path = output_dir / "screenshots" / "named_capture.png"
    screenshot_bytes = b"workspace-png"
    screenshot_path.write_bytes(screenshot_bytes)

    (output_dir / "debug" / "steps" / "step_0001.json").write_text(
        json.dumps(
            {
                "step": 1,
                "thought": "Explore the page.",
                "python_code": "python explore.py",
                "done": False,
                "outputs": [
                    {
                        "observation": {
                            "screenshot_path": str(screenshot_path),
                            "recent_screenshots": ["screenshots/named_capture.png"],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifacts = export_online_mind2web_artifacts(
        output_dir=output_dir,
        task="Example task",
        task_id="task-1",
        start_url="https://example.com",
        agent_result={"final_response": "Done", "exit_status": "Submitted", "submission": "Done"},
    )

    exported_image = output_dir / "trajectory" / "0_full_screenshot.png"
    assert artifacts["trajectory_dir"] == str(output_dir / "trajectory")
    assert exported_image.exists()
    assert exported_image.read_bytes() == screenshot_bytes


def test_export_online_mind2web_artifacts_prefers_final_script_log_for_action_history(tmp_path) -> None:
    output_dir = tmp_path / "task"
    (output_dir / "debug" / "steps").mkdir(parents=True)
    (output_dir / "screenshots").mkdir(parents=True)

    (output_dir / "debug" / "steps" / "step_0001.json").write_text(
        json.dumps(
            {
                "step": 1,
                "thought": "Explore filters.",
                "python_code": "python explore.py",
                "done": False,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "screenshots" / "step_0001.png").write_bytes(b"png")
    (output_dir / "final_script_log.txt").write_text(
        "step 1 action: open filters\n"
        "noise that should be ignored\n"
        "step 2 action: set exact price range\n",
        encoding="utf-8",
    )

    export_online_mind2web_artifacts(
        output_dir=output_dir,
        task="Example task",
        task_id="task-1",
        start_url="https://example.com",
        agent_result={"final_response": "Done", "exit_status": "Submitted", "submission": "Done"},
    )

    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["action_history"] == [
        "step 1 action: open filters",
        "step 2 action: set exact price range",
    ]
    assert payload["action_history_source"] == "final_script_log"


def test_export_online_mind2web_artifacts_all_final_execution_uses_latest_final_run(tmp_path) -> None:
    output_dir = tmp_path / "task"
    run_dir = output_dir / "final_runs" / "run_007" / "screenshots"
    run_dir.mkdir(parents=True)
    (run_dir / "final_execution_10_done.png").write_bytes(b"ten")
    (run_dir / "final_execution_2_done.png").write_bytes(b"two")

    artifacts = export_online_mind2web_artifacts_all_final_execution(
        output_dir=output_dir,
        task="Example task",
        task_id="task-1",
        start_url="https://example.com",
        agent_result={"final_response": "Done", "exit_status": "Submitted", "submission": "Done"},
    )

    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert artifacts["trajectory_dir"] == str(output_dir / "trajectory")
    assert payload["screenshot_paths"] == [
        str(output_dir / "trajectory" / "0_full_screenshot.png"),
        str(output_dir / "trajectory" / "1_full_screenshot.png"),
    ]
    assert Path(payload["screenshot_paths"][0]).read_bytes() == b"two"
    assert Path(payload["screenshot_paths"][1]).read_bytes() == b"ten"


def test_export_online_mind2web_artifacts_all_final_execution_falls_back_to_root_screenshots(tmp_path) -> None:
    output_dir = tmp_path / "task"
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True)
    (screenshots_dir / "final_execution_3_done.png").write_bytes(b"three")
    (screenshots_dir / "final_execution_1_done.png").write_bytes(b"one")

    export_online_mind2web_artifacts_all_final_execution(
        output_dir=output_dir,
        task="Example task",
        task_id="task-1",
        start_url="https://example.com",
        agent_result={"final_response": "Done", "exit_status": "Submitted", "submission": "Done"},
    )

    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert Path(payload["screenshot_paths"][0]).read_bytes() == b"one"
    assert Path(payload["screenshot_paths"][1]).read_bytes() == b"three"


def test_normalize_online_mind2web_judge_results_rewrites_missing_history_prompt(tmp_path) -> None:
    result_file = tmp_path / "judge_results.json"
    result_file.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "task": "Example task",
                "action_history": [],
                "exit_status": "RuntimeError",
                "run_exception": "Page did not finish rendering.",
                "evaluation_details": {
                    "response": "I'm sorry, but it seems like the action history is missing. Could you please provide the action history so I can evaluate the agent's performance?",
                    "predicted_label": 0,
                },
                "predicted_label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    replacements = normalize_online_mind2web_judge_results(result_file=result_file)

    assert replacements == 1
    payload = json.loads(result_file.read_text(encoding="utf-8").strip())
    assert payload["predicted_label"] == 0
    assert payload["evaluation_details"]["predicted_label"] == 0
    assert payload["evaluation_details"]["response"] == (
        "Thoughts: The action history is empty, so there is no evidence that the agent completed the task."
        " The run ended before any browser actions were recorded (exit status RuntimeError; run exception: Page did not finish rendering.)."
        ' Under the evaluation criteria, this must be marked as failure.\nStatus: "failure"'
    )


def test_normalize_online_mind2web_judge_results_leaves_valid_failure_unchanged(tmp_path) -> None:
    result_file = tmp_path / "judge_results.json"
    original_response = (
        "Thoughts: There is no action history, so the task cannot be verified.\n"
        'Status: "failure"'
    )
    result_file.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "task": "Example task",
                "action_history": [],
                "evaluation_details": {
                    "response": original_response,
                    "predicted_label": 0,
                },
                "predicted_label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    replacements = normalize_online_mind2web_judge_results(result_file=result_file)

    assert replacements == 0
    payload = json.loads(result_file.read_text(encoding="utf-8").strip())
    assert payload["evaluation_details"]["response"] == original_response


def test_run_online_mind2web_judge_defaults_to_sandbox_eval(tmp_path, monkeypatch) -> None:
    from miniswewebagent.utils import om2w_eval

    captured = {}

    def fake_run(cmd, text, capture_output):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(om2w_eval.subprocess, "run", fake_run)

    om2w_eval.run_online_mind2web_judge(
        judge_python=tmp_path / "python",
        judge_script=tmp_path / "run.py",
        trajectories_dir=tmp_path / "traj",
        output_dir=tmp_path / "out",
        judge_model="o4-mini",
        num_proc=4,
        api_key="key",
    )

    cmd = captured["cmd"]
    assert cmd[2] == "--mode"
    assert cmd[3] == "WebJudge_Online_Mind2Web_Sandbox_eval"
