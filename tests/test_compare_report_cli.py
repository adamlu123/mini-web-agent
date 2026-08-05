from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typer.testing import CliRunner

from miniswewebagent.run.utilities.compare_report import app


def _write_task(run_dir: Path, task_id: str) -> None:
    task_dir = run_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(json.dumps({"task_id": task_id, "task": "Do a thing."}), encoding="utf-8")


def _write_judge_file(run_dir: Path, name: str, rows: list[dict]) -> None:
    lines = "\n".join(json.dumps(row) for row in rows)
    (run_dir / name).write_text(lines + "\n", encoding="utf-8")


def test_compare_report_cli_prints_leaderboard_and_diff_tables(tmp_path: Path) -> None:
    runs_root = tmp_path / "outputs"
    _write_task(runs_root / "baseline", "task-1")
    _write_task(runs_root / "candidate", "task-1")
    _write_judge_file(
        runs_root / "baseline",
        "WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json",
        [{"task_id": "task-1", "predicted_label": 0}],
    )
    _write_judge_file(
        runs_root / "candidate",
        "WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json",
        [{"task_id": "task-1", "predicted_label": 1}],
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--runs", "baseline,candidate", "--runs-root", str(runs_root)])

    assert result.exit_code == 0, result.output
    assert "baseline" in result.output
    assert "candidate" in result.output
    assert "Improved" in result.output


def test_compare_report_cli_json_output(tmp_path: Path) -> None:
    runs_root = tmp_path / "outputs"
    _write_task(runs_root / "run_a", "task-1")
    _write_task(runs_root / "run_b", "task-1")

    runner = CliRunner()
    result = runner.invoke(app, ["--runs", "run_a,run_b", "--runs-root", str(runs_root), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["runIds"] == ["run_a", "run_b"]
    assert payload["baselineId"] == "run_a"


def test_compare_report_cli_rejects_unknown_run_id(tmp_path: Path) -> None:
    runs_root = tmp_path / "outputs"
    _write_task(runs_root / "run_a", "task-1")

    runner = CliRunner()
    result = runner.invoke(app, ["--runs", "run_a,does-not-exist", "--runs-root", str(runs_root)])

    assert result.exit_code != 0
    assert "does-not-exist" in result.output


def test_compare_report_cli_rejects_missing_runs_root(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--runs", "a,b", "--runs-root", str(tmp_path / "nope")])

    assert result.exit_code != 0
