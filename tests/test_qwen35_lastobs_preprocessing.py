from __future__ import annotations

import json
from pathlib import Path

from scripts.data.qwen35.preprocess_lastobs_singleturn import (
    ConversionStats,
    build_singleturn_rows,
    load_eval_success_filter,
    preprocess,
)


def _assistant(thought: str, bash: str, *, done: bool = False, final: str = "") -> dict:
    return {
        "role": "assistant",
        "content": thought,
        "extra": {
            "actions": [{"bash_command": bash}] if bash else [],
            "done": done,
            "final_response": final,
            "raw_response": {
                "thought": thought,
                "bash_command": bash,
                "python_code": "",
                "done": done,
                "final_response": final,
            },
        },
    }


def _trajectory(workspace: Path, *, status: str = "Submitted") -> dict:
    return {
        "trajectory_format": "mini-swe-webagent-0.1",
        "compacted_sessions": [],
        "environment": {
            "workspace_dir": str(workspace),
            "config": {"output_dir": str(workspace)},
        },
        "info": {"exit_status": status},
        "messages": [
            {"role": "system", "content": "SPB system", "extra": {}},
            {
                "role": "user",
                "content": f"Task: test\nTask ID: task-1\nWorkspace: {workspace}",
                "extra": {},
            },
            _assistant("first thought", "python -m browser_session create"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Observation:\nStatus: error\nCommand output:\n"
                            "/bin/bash: python: command not found\n"
                        ),
                    }
                ],
                "extra": {"observation": {"success": False}},
            },
            _assistant(
                "recover with the real interpreter",
                f"{workspace}/.venv/bin/python -m browser_session create",
            ),
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Observation:\nStatus: ok\nCommand output:\n"
                            "page contains literal <image> and sk-abcdefghijklmnop\n"
                        ),
                    }
                ],
                "extra": {"observation": {"success": True}},
            },
            {
                "role": "user",
                "content": "Format error:\nretry with a valid unified state.",
                "extra": {"interrupt_type": "FormatError"},
            },
            _assistant("finished", "", done=True, final="done"),
            {
                "role": "exit",
                "content": "done",
                "extra": {"exit_status": status, "submission": "done"},
            },
        ],
    }


def test_lastobs_singleturn_keeps_history_states_and_only_latest_user_block(tmp_path: Path) -> None:
    workspace = tmp_path / "task-1"
    trajectory_path = workspace / "trajectory.json"
    stats = ConversionStats()

    rows = build_singleturn_rows(
        _trajectory(workspace),
        trajectory_path,
        path_mode="workspace",
        stats=stats,
    )

    assert len(rows) == 3
    assert [len(row["conversations"]) for row in rows] == [2, 4, 6]

    final_conversation = rows[-1]["conversations"]
    assert [turn["from"] for turn in final_conversation] == [
        "human",
        "gpt",
        "human",
        "gpt",
        "human",
        "gpt",
    ]
    assert final_conversation[1]["value"] == (
        "<think>\nfirst thought\n</think>\n"
        "<bash>\npython -m browser_session create\n</bash>\n"
        "<done>false</done>\n"
        "<final_response>\n\n</final_response>"
    )
    assert "Command output: (omitted)" in final_conversation[2]["value"]
    assert "python: command not found" not in final_conversation[2]["value"]
    assert "page contains literal &lt;image&gt;" in final_conversation[4]["value"]
    assert "<REDACTED_SECRET>" in final_conversation[4]["value"]
    assert "Format error:" in final_conversation[4]["value"]
    assert final_conversation[-1]["value"].endswith(
        "<done>true</done>\n<final_response>\ndone\n</final_response>"
    )

    # The second target's immediately preceding observation stays complete.
    assert "python: command not found" in rows[1]["conversations"][2]["value"]
    assert str(workspace) not in json.dumps(rows)
    assert "/workspace/.venv/bin/python" in final_conversation[3]["value"]
    assert rows[-1]["metadata"]["assistant_mask_mode"] == "last"
    assert stats.historical_observations_stubbed == 1
    assert stats.adjacent_user_blocks_merged == 1
    assert stats.sanitization.secret_redactions == 1
    assert stats.sanitization.escaped_image_sentinels == 1


def test_preprocess_filters_non_submitted_and_writes_bundle(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "bundle"
    submitted = source_dir / "task-1"
    failed = source_dir / "task-2"
    submitted.mkdir(parents=True)
    failed.mkdir(parents=True)
    (submitted / "trajectory.json").write_text(
        json.dumps(_trajectory(submitted)),
        encoding="utf-8",
    )
    failed_trajectory = _trajectory(failed, status="LimitsExceeded")
    failed_trajectory["messages"][-1]["extra"]["exit_status"] = "LimitsExceeded"
    (failed / "trajectory.json").write_text(
        json.dumps(failed_trajectory),
        encoding="utf-8",
    )

    manifest = preprocess(
        source_dir,
        output_dir,
        dataset_name="test_lastobs",
        required_exit_status="Submitted",
        path_mode="workspace",
    )

    assert manifest["counts"]["source_trajectories"] == 2
    assert manifest["counts"]["selected_trajectories"] == 1
    assert manifest["counts"]["output_rows"] == 3
    assert manifest["settings"]["assistant_mask_mode_required"] == "last"
    assert (output_dir / "dataset_info.json").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "source_tasks.json").is_file()
    rows = [
        json.loads(line)
        for line in (output_dir / "test_lastobs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    assert rows[-1]["id"] == "task-1-step-0003"


def test_preprocess_intersects_submitted_with_eval_success(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "bundle"
    for task_id in ("task-1", "task-2"):
        task_dir = source_dir / task_id
        task_dir.mkdir(parents=True)
        trajectory = _trajectory(task_dir)
        trajectory["messages"][1]["content"] = (
            f"Task: test\nTask ID: {task_id}\nWorkspace: {task_dir}"
        )
        (task_dir / "trajectory.json").write_text(
            json.dumps(trajectory),
            encoding="utf-8",
        )

    result_path = tmp_path / "judge-results.json"
    result_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "task-1", "predicted_label": 1}),
                json.dumps({"task_id": "task-2", "predicted_label": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "eval_summary.json"
    summary_path.write_text(
        json.dumps({"result_file": str(result_path), "successful_tasks": 1}),
        encoding="utf-8",
    )
    eval_filter = load_eval_success_filter(summary_path)

    manifest = preprocess(
        source_dir,
        output_dir,
        dataset_name="success_only",
        required_exit_status="Submitted",
        path_mode="workspace",
        eval_filter=eval_filter,
    )

    assert manifest["counts"]["selected_trajectories"] == 1
    assert manifest["counts"]["output_rows"] == 3
    assert manifest["counts"]["skipped_reasons"] == {"eval_label_not_1": 1}
    assert manifest["settings"]["eval_success_filter"]["successful_tasks"] == 1
    rows = [
        json.loads(line)
        for line in (output_dir / "success_only.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["metadata"]["task_id"] for row in rows} == {"task-1"}
