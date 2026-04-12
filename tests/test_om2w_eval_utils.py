from __future__ import annotations

import json
import subprocess
from pathlib import Path

from miniswewebagent.utils.om2w_eval import export_online_mind2web_artifacts, normalize_online_mind2web_judge_results


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
    assert payload["final_artifact_dir"] == str(output_dir)
    assert len(payload["screenshot_paths"]) == 2

    exported_image = output_dir / "trajectory" / "0_full_screenshot.png"
    placeholder_image = output_dir / "trajectory" / "1_full_screenshot.png"
    assert artifacts["trajectory_dir"] == str(output_dir / "trajectory")
    assert exported_image.exists()
    assert placeholder_image.exists()
    assert placeholder_image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_export_online_mind2web_artifacts_uses_command_text_for_bash_debug_steps(tmp_path) -> None:
    output_dir = tmp_path / "task"
    (output_dir / "debug" / "steps").mkdir(parents=True)
    (output_dir / "screenshots").mkdir(parents=True)

    (output_dir / "debug" / "steps" / "step_0001.json").write_text(
        json.dumps(
            {
                "step": 1,
                "thought": "Open the homepage.",
                "python_code": "",
                "bash_command": "python script.py",
                "command_text": "python script.py",
                "done": False,
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

    payload = json.loads(Path(artifacts["result_json"]).read_text(encoding="utf-8"))
    assert payload["action_history"] == ["python script.py"]


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


def test_export_online_mind2web_artifacts_prefers_matching_final_run_screenshot_for_verification_steps(tmp_path) -> None:
    output_dir = tmp_path / "task"
    (output_dir / "debug" / "steps").mkdir(parents=True)
    (output_dir / "screenshots").mkdir(parents=True)

    stale_screenshot = output_dir / "screenshots" / "state_5_after_destination.png"
    stale_screenshot.write_bytes(b"stale")

    final_run_dir = output_dir / "final_runs" / "run_007"
    final_run_screenshots = final_run_dir / "screenshots"
    final_run_screenshots.mkdir(parents=True)
    final_run_screenshot = final_run_screenshots / "final_execution_11_results_dates_reapplied.png"
    final_run_screenshot.write_bytes(b"artifact")

    (output_dir / "debug" / "steps" / "step_0001.json").write_text(
        json.dumps(
            {
                "step": 1,
                "thought": "Inspect the latest final run verification artifacts.",
                "python_code": "python - <<'PY'\nprint('step_11_results_dates_reapplied.aria.txt')\nPY",
                "done": False,
                "outputs": [
                    {
                        "observation": {
                            "screenshot_path": str(stale_screenshot),
                            "recent_screenshots": ["screenshots/state_5_after_destination.png"],
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
    assert exported_image.read_bytes() == b"artifact"


def test_export_online_mind2web_artifacts_keeps_nonstale_observation_screenshot_for_verification_steps(tmp_path) -> None:
    output_dir = tmp_path / "task"
    (output_dir / "debug" / "steps").mkdir(parents=True)
    (output_dir / "screenshots").mkdir(parents=True)

    current_screenshot = output_dir / "screenshots" / "15_review_sort_select_probe.png"
    current_screenshot.write_bytes(b"current")

    final_run_dir = output_dir / "final_runs" / "run_007"
    final_run_screenshots = final_run_dir / "screenshots"
    final_run_screenshots.mkdir(parents=True)
    (final_run_screenshots / "final_execution_11_results_dates_reapplied.png").write_bytes(b"artifact")

    (output_dir / "debug" / "steps" / "step_0001.json").write_text(
        json.dumps(
            {
                "step": 1,
                "thought": "Inspect the latest final run verification artifacts.",
                "python_code": "python - <<'PY'\nprint('step_11_results_dates_reapplied.aria.txt')\nPY",
                "done": False,
                "outputs": [
                    {
                        "observation": {
                            "screenshot_path": str(current_screenshot),
                            "recent_screenshots": ["screenshots/15_review_sort_select_probe.png"],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    export_online_mind2web_artifacts(
        output_dir=output_dir,
        task="Example task",
        task_id="task-1",
        start_url="https://example.com",
        agent_result={"final_response": "Done", "exit_status": "Submitted", "submission": "Done"},
    )

    exported_image = output_dir / "trajectory" / "0_full_screenshot.png"
    assert exported_image.exists()
    assert exported_image.read_bytes() == b"current"


def test_export_online_mind2web_artifacts_uses_placeholder_when_step_screenshot_is_missing(tmp_path) -> None:
    output_dir = tmp_path / "task"
    (output_dir / "steps").mkdir(parents=True)

    (output_dir / "steps" / "step_0001.py").write_text("print('workspace run ok')\n", encoding="utf-8")
    (output_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Run the workspace script.",
                        "extra": {
                            "actions": [
                                {
                                    "bash_command": "python final_script.py",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    artifacts = export_online_mind2web_artifacts(
        output_dir=output_dir,
        task="Workspace task",
        task_id="task-1",
        start_url="https://example.com",
        agent_result={"final_response": "Done", "exit_status": "Submitted", "submission": "Done"},
    )

    exported_image = output_dir / "trajectory" / "0_full_screenshot.png"
    assert artifacts["trajectory_dir"] == str(output_dir / "trajectory")
    assert exported_image.exists()
    assert exported_image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["action_history"] == ["python final_script.py"]
    assert payload["action_history_source"] == "trajectory"
    assert payload["screenshot_paths"] == [str(exported_image)]


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
    assert payload["final_artifact_dir"] == str(output_dir)


def test_export_online_mind2web_artifacts_prefers_latest_final_run_folder_for_action_history(tmp_path) -> None:
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
        "step 1 action: legacy root action\n",
        encoding="utf-8",
    )
    run_dir = output_dir / "final_runs" / "run_007"
    run_dir.mkdir(parents=True)
    (run_dir / "final_script_log.txt").write_text(
        "step 1 action: newer folder action\n"
        "step 2 action: final confirmation\n",
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
        "step 1 action: newer folder action",
        "step 2 action: final confirmation",
    ]
    assert payload["action_history_source"] == "final_script_log"
    assert payload["final_artifact_dir"] == str(run_dir)


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
    assert "--endpoint_target_uri" not in cmd


def test_run_online_mind2web_judge_passes_gateway_endpoint(tmp_path, monkeypatch) -> None:
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
        judge_model="gpt-5.4",
        num_proc=4,
        api_key="gateway-key",
        endpoint_target_uri="http://gateway.example/api/responses",
    )

    cmd = captured["cmd"]
    endpoint_index = cmd.index("--endpoint_target_uri")
    assert cmd[endpoint_index + 1] == "http://gateway.example/api/responses"
