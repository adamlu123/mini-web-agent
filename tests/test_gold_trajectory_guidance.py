from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniswewebagent.run.mini import _load_gold_trajectory_guidance


def _write_gold_task(root: Path, task_id: str, *, n_steps: int = 2) -> Path:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "task": "Complete the matching task.",
                "start_url": "https://example.com/",
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "plan.md").write_text("1. Inspect.\n2. Verify.\n", encoding="utf-8")
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "action_history": [f"Reference action {index}" for index in range(1, n_steps + 1)],
                "thoughts": [f"Reasoning {index} " + ("detail " * 100) for index in range(1, n_steps + 1)],
                "final_result_response": "Final Response: verified",
                "exit_status": "Submitted",
            }
        ),
        encoding="utf-8",
    )
    return task_dir


def test_gold_guidance_selects_exact_task_and_preserves_route(tmp_path: Path) -> None:
    _write_gold_task(tmp_path, "task-1")
    _write_gold_task(tmp_path, "task-2")

    guidance = _load_gold_trajectory_guidance(tmp_path, "task-2")

    assert "Source task ID: task-2" in guidance
    assert "Step 1:" in guidance
    assert "Action: Reference action 1" in guidance
    assert "Action: Reference action 2" in guidance
    assert "Final Response: verified" in guidance
    assert "task-1" not in guidance


def test_gold_guidance_bounds_reasoning_without_dropping_actions(tmp_path: Path) -> None:
    _write_gold_task(tmp_path, "task-1", n_steps=12)

    guidance = _load_gold_trajectory_guidance(tmp_path, "task-1", max_chars=3_000)

    assert len(guidance) <= 3_000
    for index in range(1, 13):
        assert f"Action: Reference action {index}" in guidance
    assert "Final Response: verified" in guidance


def test_gold_guidance_fails_loudly_when_task_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="task-missing"):
        _load_gold_trajectory_guidance(tmp_path, "task-missing")
