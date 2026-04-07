from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miniswewebagent.run.utilities.review_viewer import ReviewCatalog


def test_review_catalog_lists_runs_judges_tasks_and_reasons(tmp_path) -> None:
    runs_root = tmp_path / "outputs" / "sandbox"
    judge_root = tmp_path / "om2w_judge"

    task_dir = runs_root / "run_alpha" / "task_folder_001"
    (task_dir / "screenshots" / "nested").mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-001",
                "task": "Find the right item.",
                "start_url": "https://example.com",
                "exit_status": "Submitted",
                "thoughts": ["open page", "apply filter"],
                "action_history": ["goto", "click filter"],
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "screenshots" / "final_execution_10_done.png").write_bytes(b"png")
    (task_dir / "screenshots" / "final_execution_2_done.png").write_bytes(b"png")
    (task_dir / "screenshots" / "nested" / "extra.png").write_bytes(b"png")

    summary_dir = judge_root / "output_run_alpha"
    summary_dir.mkdir(parents=True)
    (summary_dir / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# Summary",
                "",
                "| Task ID | Status | Judge Reason |",
                "| --- | --- | --- |",
                "| `task-001` | `failure` | Missing the exact filter confirmation. |",
            ]
        ),
        encoding="utf-8",
    )
    (summary_dir / "WebJudge_Online_Mind2Web_Sandbox_eval_o4-mini_score_threshold_3_auto_eval_results.json").write_text(
        json.dumps(
            {
                "task_id": "task-001",
                "predicted_label": 0,
                "evaluation_details": {
                    "response": "Thoughts: The final page did not confirm the required filter.\nStatus: failure"
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    catalog = ReviewCatalog(runs_root, judge_root)

    runs = catalog.list_runs()
    assert [run["id"] for run in runs] == ["run_alpha"]

    judge_files = catalog.list_judge_files()
    judge_ids = {entry["id"] for entry in judge_files}
    assert "output_run_alpha/SUMMARY.md" in judge_ids
    assert (
        "output_run_alpha/WebJudge_Online_Mind2Web_Sandbox_eval_o4-mini_score_threshold_3_auto_eval_results.json"
        in judge_ids
    )

    tasks = catalog.list_tasks("run_alpha", "output_run_alpha/SUMMARY.md")
    assert len(tasks) == 1
    assert tasks[0]["taskId"] == "task-001"
    assert tasks[0]["judgeStatus"] == "failure"
    assert tasks[0]["imageCount"] == 2

    detail = catalog.task_detail("run_alpha", "task_folder_001", "output_run_alpha/SUMMARY.md")
    assert detail["taskId"] == "task-001"
    assert detail["task"] == "Find the right item."
    assert [image["name"] for image in detail["images"]] == [
        "final_execution_2_done.png",
        "final_execution_10_done.png",
    ]
    assert detail["judge"]["status"] == "failure"
    assert detail["judge"]["reason"] == "Missing the exact filter confirmation."
    assert detail["exitStatus"] == "Submitted"
    assert detail["lastThought"] == "apply filter"
    assert detail["lastAction"] == "click filter"

    json_detail = catalog.task_detail(
        "run_alpha",
        "task_folder_001",
        "output_run_alpha/WebJudge_Online_Mind2Web_Sandbox_eval_o4-mini_score_threshold_3_auto_eval_results.json",
    )
    assert json_detail["judge"]["reason"] == "Missing the exact filter confirmation."
    assert json_detail["judge"]["response"] == (
        "Thoughts: The final page did not confirm the required filter.\nStatus: failure"
    )
