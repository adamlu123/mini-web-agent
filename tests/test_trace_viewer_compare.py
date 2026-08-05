from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miniswewebagent.run.utilities.trace_viewer import build_server


def _write_task(run_dir: Path, task_id: str, *, level_hint: str = "") -> None:
    task_dir = run_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(
        json.dumps({"task_id": task_id, "task": f"Task {task_id} ({level_hint})"}), encoding="utf-8"
    )


def _write_judge_file(run_dir: Path, name: str, rows: list[dict]) -> None:
    lines = "\n".join(json.dumps(row) for row in rows)
    (run_dir / name).write_text(lines + "\n", encoding="utf-8")


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _start_server(root: Path):
    server = build_server(root, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_api_compare_returns_leaderboard_and_task_diff(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    baseline_dir = root / "baseline"
    candidate_dir = root / "candidate"

    _write_task(baseline_dir, "task-1")
    _write_task(baseline_dir, "task-2")
    _write_task(candidate_dir, "task-1")
    _write_task(candidate_dir, "task-2")

    _write_judge_file(
        baseline_dir,
        "WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json",
        [
            {"task_id": "task-1", "predicted_label": 0},
            {"task_id": "task-2", "predicted_label": 1},
        ],
    )
    _write_judge_file(
        candidate_dir,
        "WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json",
        [
            {"task_id": "task-1", "predicted_label": 1},
            {"task_id": "task-2", "predicted_label": 1},
        ],
    )

    server = _start_server(root)
    try:
        payload = _get_json(
            f"http://127.0.0.1:{server.server_port}/api/compare?runs=baseline,candidate&baseline=baseline"
        )
    finally:
        server.shutdown()
        server.server_close()

    assert payload["runIds"] == ["baseline", "candidate"]
    assert payload["baselineId"] == "baseline"
    assert [row["runId"] for row in payload["leaderboard"]] == ["baseline", "candidate"]

    tasks_by_id = {task["taskId"]: task for task in payload["tasks"]}
    assert tasks_by_id["task-1"]["flipsVsBaseline"]["candidate"] == "improved"
    assert tasks_by_id["task-2"]["flipsVsBaseline"]["candidate"] == "same_success"


def test_api_compare_defaults_baseline_to_first_run(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_task(root / "run_a", "task-1")
    _write_task(root / "run_b", "task-1")

    server = _start_server(root)
    try:
        payload = _get_json(f"http://127.0.0.1:{server.server_port}/api/compare?runs=run_a,run_b")
    finally:
        server.shutdown()
        server.server_close()

    assert payload["baselineId"] == "run_a"


def test_api_compare_rejects_unknown_run_id(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_task(root / "run_a", "task-1")

    server = _start_server(root)
    try:
        try:
            _get_json(f"http://127.0.0.1:{server.server_port}/api/compare?runs=run_a,does-not-exist")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("Expected a 404 for an unknown run id")
    finally:
        server.shutdown()
        server.server_close()


def test_api_compare_rejects_baseline_not_in_runs(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_task(root / "run_a", "task-1")
    _write_task(root / "run_b", "task-1")

    server = _start_server(root)
    try:
        try:
            _get_json(f"http://127.0.0.1:{server.server_port}/api/compare?runs=run_a,run_b&baseline=run_c")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("Expected a 400 when baseline is not one of runs")
    finally:
        server.shutdown()
        server.server_close()


def test_api_compare_uses_bundled_task_levels(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    known_task_id = "b7258ee05d75e6c50673a59914db412e_110325"  # bundled om2w_260220.json sample, level=medium
    _write_task(root / "run_a", known_task_id)
    _write_task(root / "run_b", known_task_id)

    server = _start_server(root)
    try:
        payload = _get_json(f"http://127.0.0.1:{server.server_port}/api/compare?runs=run_a,run_b")
    finally:
        server.shutdown()
        server.server_close()

    tasks_by_id = {task["taskId"]: task for task in payload["tasks"]}
    assert tasks_by_id[known_task_id]["level"] == "medium"
