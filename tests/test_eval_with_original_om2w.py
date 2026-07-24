from pathlib import Path

from scripts.eval_with_original_om2w import bound_action_history, load_action_history, load_step_actions


def test_bound_action_history_preserves_small_history() -> None:
    actions = ["open page", "apply filter", "verify results"]

    assert bound_action_history(actions) is actions


def test_bound_action_history_keeps_both_ends_within_limits() -> None:
    actions = [f"action-{index}-" + ("x" * 500) for index in range(1_000)]

    bounded = bound_action_history(actions)

    assert len(bounded) == 501
    assert bounded[0].startswith("action-0-")
    assert "500 action log line(s) omitted" in bounded[250]
    assert bounded[-1].startswith("action-999-")
    assert sum(map(len, bounded)) <= 60_000


def test_load_step_actions_uses_numeric_order(tmp_path: Path) -> None:
    steps_dir = tmp_path / "steps"
    steps_dir.mkdir()
    (steps_dir / "step_10.sh").write_text("click 10\nconfirm", encoding="utf-8")
    (steps_dir / "step_2.sh").write_text("click 2", encoding="utf-8")
    (steps_dir / "step_1.sh").write_text("click 1", encoding="utf-8")
    (steps_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    assert load_step_actions(steps_dir) == ["click 1", "click 2", "click 10\nconfirm"]


def test_step_action_source_does_not_read_final_script_log(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    run_dir = task_dir / "final_runs" / "run_4"
    (task_dir / "steps").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (task_dir / "steps" / "3.sh").write_text("step action", encoding="utf-8")
    (run_dir / "final_script_log.txt").write_text("wrong source", encoding="utf-8")

    assert load_action_history(task_dir, run_dir, "step_scripts") == ["step action"]
