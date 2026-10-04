from __future__ import annotations

import argparse
import json
from dataclasses import replace

import pytest

from miniswewebagent.evaluation.om2w import runner
from miniswewebagent.evaluation.om2w.artifacts import DEFAULT_ARTIFACT_SPEC, TaskArtifacts
from om2w_judge.utils import extract_predication, parse_status_verdict


def valid_row(task_id="task", label=1):
    verdict = "success" if label else "failure"
    return {
        "task_id": task_id,
        "predicted_label": label,
        "evaluation_details": {"response": f"Thoughts: evidence\nStatus: {verdict}"},
        "action_history_source": DEFAULT_ARTIFACT_SPEC.action_history_source,
        "screenshot_source": DEFAULT_ARTIFACT_SPEC.screenshot_source,
        "screenshot_paths": [],
        "image_judge_record": [],
    }


@pytest.mark.parametrize(
    ("response", "label"),
    [
        ('Thoughts: complete\nStatus: "success"', 1),
        ("Thoughts: incomplete\nStatus: failure", 0),
        ("Status: \u201csuccess\u201d", 1),
        ("Status: failure\nCorrection. Status: SUCCESS", 1),
    ],
)
def test_parse_status_verdict(response, label):
    assert parse_status_verdict(response) == label


@pytest.mark.parametrize("response", ["", "429", "Reasoning: evidence\nScore: 2", None])
def test_invalid_verdict_is_not_a_task_failure(response):
    with pytest.raises(ValueError, match="no valid Status"):
        parse_status_verdict(response)
    assert extract_predication(response, runner.MODE) == 0


def test_resume_uses_last_row_and_retries_invalid_judgments(tmp_path):
    path = tmp_path / "results.json"
    malformed = valid_row("malformed", 0)
    malformed["evaluation_details"]["response"] = "Reasoning: evidence\nScore: 2"
    incomplete = valid_row("incomplete")
    incomplete["screenshot_paths"] = ["shot.png"]
    incomplete["image_judge_record"] = [
        {"Score": 0, "Reasoning": "", "ParseFailed": True, "ParseError": "429"}
    ]
    rows = [
        valid_row("valid-success"),
        valid_row("valid-failure", 0),
        valid_row("latest-error"),
        {"task_id": "latest-error", "evaluation_error": "429"},
        malformed,
        incomplete,
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert runner.completed_task_ids(path) == {"valid-success", "valid-failure"}
    summary = runner.summarize_results(path, {row["task_id"] for row in rows})
    assert summary["completed_tasks"] == 2
    assert summary["error_tasks"] == 3
    assert summary["successful_tasks"] == 1


def test_resume_rejects_layout_change_before_overwriting_manifest(tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    manifest_path = output / "eval_manifest.json"
    manifest_path.write_text('{"original": true}')
    row = valid_row()
    row["action_history_source"] = "steps/step_<id>.sh"
    results_path = runner.output_results_path(output, "o4-mini", 3)
    results_path.write_text(json.dumps(row) + "\n")
    artifact = TaskArtifacts("task", str(tmp_path), "Task", ["Action"], [])
    spec = replace(DEFAULT_ARTIFACT_SPEC, loader=lambda _: [artifact])
    args = argparse.Namespace(
        trajectories_dir=str(tmp_path),
        output_path=str(output),
        expected_tasks=1,
        model="o4-mini",
        score_threshold=3,
    )

    with pytest.raises(ValueError, match="different artifact layout"):
        runner.parallel_eval(args, spec=spec)
    assert manifest_path.read_text() == '{"original": true}'
