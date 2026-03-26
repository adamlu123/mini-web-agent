from __future__ import annotations

import json

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

    exported_image = output_dir / "trajectory" / "0_full_screenshot.png"
    assert artifacts["trajectory_dir"] == str(output_dir / "trajectory")
    assert exported_image.exists()


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
