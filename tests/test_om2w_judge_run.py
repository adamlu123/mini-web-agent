import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from om2w_judge.run import load_final_script_action_history


def test_load_final_script_action_history_reads_only_step_lines(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "final_script_log.txt").write_text(
        "step 1 action: search destination\n"
        "FINAL_URL: https://example.com\n"
        "step 2 action: apply exact filters\n",
        encoding="utf-8",
    )

    assert load_final_script_action_history(task_dir) == [
        "step 1 action: search destination",
        "step 2 action: apply exact filters",
    ]
