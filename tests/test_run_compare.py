from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from miniswewebagent.run.utilities.run_compare import (
    FLIP_IMPROVED,
    FLIP_REGRESSED,
    FLIP_SAME_FAIL,
    FLIP_SAME_SUCCESS,
    FLIP_UNKNOWN,
    STATUS_FAILURE,
    STATUS_MISSING,
    STATUS_SUCCESS,
    STATUS_UNKNOWN,
    classify_flip,
    compare_runs,
    load_run_task_statuses,
    summarize_run,
)


def _write_task(
    run_dir: Path, task_id: str, *, task: str = "Do a thing.", folder_name: str | None = None
) -> None:
    task_dir = run_dir / (folder_name or task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(
        json.dumps({"task_id": task_id, "task": task}), encoding="utf-8"
    )


def _write_judge_file(run_dir: Path, name: str, rows: list[dict]) -> None:
    lines = "\n".join(json.dumps(row) for row in rows)
    (run_dir / name).write_text(lines + "\n", encoding="utf-8")


class TestLoadRunTaskStatuses:
    def test_missing_run_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_run_task_statuses(tmp_path / "nope") == {}

    def test_task_without_judge_row_is_unknown(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _write_task(run_dir, "task-1", task="Search for a thing.")

        statuses = load_run_task_statuses(run_dir)

        assert statuses == {
            "task-1": {"title": "Search for a thing.", "status": STATUS_UNKNOWN, "folderName": "task-1"}
        }

    def test_single_judge_file_resolves_success_and_failure(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _write_task(run_dir, "task-1")
        _write_task(run_dir, "task-2")
        _write_judge_file(
            run_dir,
            "WebJudge_Online_Mind2Web_eval_gpt4o_score_threshold_3_auto_eval_results.json",
            [
                {"task_id": "task-1", "predicted_label": 1},
                {"task_id": "task-2", "predicted_label": 0},
            ],
        )

        statuses = load_run_task_statuses(run_dir)

        assert statuses["task-1"]["status"] == STATUS_SUCCESS
        assert statuses["task-2"]["status"] == STATUS_FAILURE

    def test_folder_name_is_tracked_separately_from_task_id(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _write_task(run_dir, "task-1", folder_name="0_some_other_dirname")

        statuses = load_run_task_statuses(run_dir)

        assert statuses["task-1"]["folderName"] == "0_some_other_dirname"

    def test_judge_only_task_falls_back_to_task_id_as_folder_name(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        run_dir.mkdir(parents=True)
        _write_judge_file(run_dir, "WebJudge_eval_1_auto_eval_results.json", [{"task_id": "orphan-task", "predicted_label": 1}])

        statuses = load_run_task_statuses(run_dir)

        assert statuses["orphan-task"]["folderName"] == "orphan-task"


    def test_majority_vote_across_multiple_judge_files(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _write_task(run_dir, "task-1")
        _write_judge_file(run_dir, "WebJudge_eval_1_auto_eval_results.json", [{"task_id": "task-1", "predicted_label": 1}])
        _write_judge_file(run_dir, "WebJudge_eval_2_auto_eval_results.json", [{"task_id": "task-1", "predicted_label": 1}])
        _write_judge_file(run_dir, "WebJudge_eval_3_auto_eval_results.json", [{"task_id": "task-1", "predicted_label": 0}])

        statuses = load_run_task_statuses(run_dir)

        assert statuses["task-1"]["status"] == STATUS_SUCCESS

    def test_tied_judge_votes_are_unknown(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _write_task(run_dir, "task-1")
        _write_judge_file(run_dir, "WebJudge_eval_1_auto_eval_results.json", [{"task_id": "task-1", "predicted_label": 1}])
        _write_judge_file(run_dir, "WebJudge_eval_2_auto_eval_results.json", [{"task_id": "task-1", "predicted_label": 0}])

        statuses = load_run_task_statuses(run_dir)

        assert statuses["task-1"]["status"] == STATUS_UNKNOWN

    def test_malformed_judge_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_a"
        _write_task(run_dir, "task-1")
        judge_path = run_dir / "WebJudge_eval_1_auto_eval_results.json"
        judge_path.write_text('{"task_id": "task-1", "predicted_label": 1}\nnot json\n', encoding="utf-8")

        statuses = load_run_task_statuses(run_dir)

        assert statuses["task-1"]["status"] == STATUS_SUCCESS


class TestSummarizeRun:
    def test_overall_and_per_level_counts_with_success_rate(self) -> None:
        statuses = {
            "e1": {"title": "", "status": STATUS_SUCCESS},
            "e2": {"title": "", "status": STATUS_FAILURE},
            "h1": {"title": "", "status": STATUS_SUCCESS},
            "u1": {"title": "", "status": STATUS_UNKNOWN},
        }
        task_levels = {"e1": "easy", "e2": "easy", "h1": "hard"}

        summary = summarize_run("run_a", statuses, task_levels)

        assert summary["runId"] == "run_a"
        assert summary["totalTasks"] == 4
        assert summary["overall"] == {
            "success": 2,
            "failure": 1,
            "unknown": 1,
            "total": 4,
            "successRate": 0.5,
        }
        assert summary["byLevel"]["easy"] == {
            "success": 1,
            "failure": 1,
            "unknown": 0,
            "total": 2,
            "successRate": 0.5,
        }
        assert summary["byLevel"]["hard"]["successRate"] == 1.0
        assert summary["byLevel"]["unknown"]["total"] == 1

    def test_empty_run_has_zero_success_rate(self) -> None:
        summary = summarize_run("empty_run", {}, {})
        assert summary["totalTasks"] == 0
        assert summary["overall"]["successRate"] == 0.0


class TestClassifyFlip:
    @pytest.mark.parametrize(
        ("baseline", "candidate", "expected"),
        [
            (STATUS_SUCCESS, STATUS_FAILURE, FLIP_REGRESSED),
            (STATUS_FAILURE, STATUS_SUCCESS, FLIP_IMPROVED),
            (STATUS_SUCCESS, STATUS_SUCCESS, FLIP_SAME_SUCCESS),
            (STATUS_FAILURE, STATUS_FAILURE, FLIP_SAME_FAIL),
            (STATUS_UNKNOWN, STATUS_SUCCESS, FLIP_UNKNOWN),
            (STATUS_SUCCESS, STATUS_MISSING, FLIP_UNKNOWN),
            (STATUS_MISSING, STATUS_MISSING, FLIP_UNKNOWN),
        ],
    )
    def test_classify_flip(self, baseline: str, candidate: str, expected: str) -> None:
        assert classify_flip(baseline, candidate) == expected


class TestCompareRuns:
    def _make_two_runs(self, tmp_path: Path) -> Path:
        root = tmp_path / "outputs"

        run_a = root / "run_a"
        _write_task(run_a, "improves", task="Task that gets fixed.")
        _write_task(run_a, "regresses", task="Task that breaks.")
        _write_task(run_a, "stays-good", task="Reliable task.")
        _write_task(run_a, "stays-bad", task="Consistently hard task.")
        _write_task(run_a, "only-in-a", task="Dropped from candidate run.")
        _write_judge_file(
            run_a,
            "WebJudge_eval_1_auto_eval_results.json",
            [
                {"task_id": "improves", "predicted_label": 0},
                {"task_id": "regresses", "predicted_label": 1},
                {"task_id": "stays-good", "predicted_label": 1},
                {"task_id": "stays-bad", "predicted_label": 0},
                {"task_id": "only-in-a", "predicted_label": 1},
            ],
        )

        run_b = root / "run_b"
        _write_task(run_b, "improves", task="Task that gets fixed.")
        _write_task(run_b, "regresses", task="Task that breaks.")
        _write_task(run_b, "stays-good", task="Reliable task.")
        _write_task(run_b, "stays-bad", task="Consistently hard task.")
        _write_task(run_b, "only-in-b", task="New task added.")
        _write_judge_file(
            run_b,
            "WebJudge_eval_1_auto_eval_results.json",
            [
                {"task_id": "improves", "predicted_label": 1},
                {"task_id": "regresses", "predicted_label": 0},
                {"task_id": "stays-good", "predicted_label": 1},
                {"task_id": "stays-bad", "predicted_label": 0},
                {"task_id": "only-in-b", "predicted_label": 1},
            ],
        )
        return root

    def test_two_run_diff_classifies_every_task(self, tmp_path: Path) -> None:
        root = self._make_two_runs(tmp_path)
        task_levels = {
            "improves": "easy",
            "regresses": "hard",
            "stays-good": "medium",
            "stays-bad": "hard",
            "only-in-a": "easy",
            "only-in-b": "medium",
        }

        result = compare_runs(root, ["run_a", "run_b"], task_levels=task_levels)

        assert result["baselineId"] == "run_a"
        assert result["runIds"] == ["run_a", "run_b"]
        assert {row["runId"] for row in result["leaderboard"]} == {"run_a", "run_b"}

        flips_by_task = {task["taskId"]: task["flipsVsBaseline"]["run_b"] for task in result["tasks"]}
        assert flips_by_task["improves"] == FLIP_IMPROVED
        assert flips_by_task["regresses"] == FLIP_REGRESSED
        assert flips_by_task["stays-good"] == FLIP_SAME_SUCCESS
        assert flips_by_task["stays-bad"] == FLIP_SAME_FAIL
        assert flips_by_task["only-in-a"] == FLIP_UNKNOWN
        assert flips_by_task["only-in-b"] == FLIP_UNKNOWN

        statuses_by_task = {task["taskId"]: task["statuses"] for task in result["tasks"]}
        assert statuses_by_task["only-in-a"]["run_b"] == STATUS_MISSING
        assert statuses_by_task["only-in-b"]["run_a"] == STATUS_MISSING

        run_dir_names_by_task = {task["taskId"]: task["runDirNames"] for task in result["tasks"]}
        assert run_dir_names_by_task["improves"] == {"run_a": "improves", "run_b": "improves"}
        # A run missing a task must not contribute a runDirNames entry for it.
        assert run_dir_names_by_task["only-in-a"] == {"run_a": "only-in-a"}

        diff = result["diffSummary"]["run_b"]
        assert diff["improved"] == 1
        assert diff["regressed"] == 1
        assert diff["sameSuccess"] == 1
        assert diff["sameFail"] == 1
        assert diff["unknown"] == 2
        assert diff["byLevel"]["hard"] == {
            "improved": 0,
            "regressed": 1,
            "sameSuccess": 0,
            "sameFail": 1,
            "unknown": 0,
        }

    def test_explicit_baseline_can_be_the_candidate_run(self, tmp_path: Path) -> None:
        root = self._make_two_runs(tmp_path)

        result = compare_runs(root, ["run_a", "run_b"], baseline_id="run_b")

        assert result["baselineId"] == "run_b"
        # Diff is now keyed by run_a (the non-baseline run) and directions flip.
        flips_by_task = {task["taskId"]: task["flipsVsBaseline"]["run_a"] for task in result["tasks"]}
        assert flips_by_task["improves"] == FLIP_REGRESSED
        assert flips_by_task["regresses"] == FLIP_IMPROVED

    def test_three_run_comparison_produces_diff_per_candidate(self, tmp_path: Path) -> None:
        root = tmp_path / "outputs"
        for run_id, label in (("run_a", 1), ("run_b", 0), ("run_c", 1)):
            run_dir = root / run_id
            _write_task(run_dir, "solo-task")
            _write_judge_file(
                run_dir,
                "WebJudge_eval_1_auto_eval_results.json",
                [{"task_id": "solo-task", "predicted_label": label}],
            )

        result = compare_runs(root, ["run_a", "run_b", "run_c"])

        assert set(result["diffSummary"].keys()) == {"run_b", "run_c"}
        task = result["tasks"][0]
        assert task["flipsVsBaseline"]["run_b"] == FLIP_REGRESSED
        assert task["flipsVsBaseline"]["run_c"] == FLIP_SAME_SUCCESS

    def test_rejects_empty_run_ids(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            compare_runs(tmp_path, [])

    def test_rejects_baseline_not_in_run_ids(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            compare_runs(tmp_path, ["run_a", "run_b"], baseline_id="run_z")
