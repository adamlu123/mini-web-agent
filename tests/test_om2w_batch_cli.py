from __future__ import annotations

import json
import subprocess

from typer.testing import CliRunner

from miniswewebagent.run.benchmarks.om2w import DEFAULT_OM2W_CONFIGS, app


def test_om2w_cli_defaults_run_without_showing_help() -> None:
    assert app.info.no_args_is_help is False


def test_om2w_cli_runs_selected_tasks(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "first",
                    "confirmed_task": "Open the first page.",
                    "website": "https://example.com/1",
                    "level": "hard",
                },
                {
                    "task_id": "second",
                    "confirmed_task": "Open the second page.",
                    "website": "https://example.com/2",
                    "level": "hard",
                },
            ]
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--task-id",
            "second",
            "--workers",
            "1",
            "--no-evaluate",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["task_id"] == "second"
    assert calls[0]["task"] == "Open the second page."
    assert calls[0]["start_url"] == "https://example.com/2"
    assert calls[0]["config_spec"] == DEFAULT_OM2W_CONFIGS
    assert calls[0]["resolved_output_dir"].name == "second"
    assert (output_dir / "config_snapshot" / "config_spec_manifest.json").exists()
    assert (output_dir / "config_snapshot" / "merged_config.yaml").exists()


def test_om2w_cli_respects_limit(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "first",
                    "confirmed_task": "Open the first page.",
                    "website": "https://example.com/1",
                    "level": "hard",
                },
                {
                    "task_id": "second",
                    "confirmed_task": "Open the second page.",
                    "website": "https://example.com/2",
                    "level": "hard",
                },
            ]
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--limit",
            "1",
            "--workers",
            "1",
            "--no-evaluate",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["task_id"] == "first"
    assert calls[0]["config_spec"] == DEFAULT_OM2W_CONFIGS
    assert calls[0]["resolved_output_dir"].name == "first"


def test_om2w_cli_filters_by_level(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "easy",
                    "confirmed_task": "Open the easy page.",
                    "website": "https://example.com/easy",
                    "level": "easy",
                },
                {
                    "task_id": "hard",
                    "confirmed_task": "Open the hard page.",
                    "website": "https://example.com/hard",
                    "level": "hard",
                },
            ]
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--task-level",
            "hard",
            "--workers",
            "1",
            "--no-evaluate",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["task_id"] == "hard"
    assert calls[0]["resolved_output_dir"].name == "hard"


def test_om2w_cli_uses_gateway_judge_endpoint_from_config(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "first",
                    "confirmed_task": "Open the first page.",
                    "website": "https://example.com/1",
                    "level": "hard",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_one(**kwargs):
        return {"final_response": "ok"}

    captured: list[dict[str, object]] = []

    def fake_run_online_mind2web_judge(**kwargs):
        captured.append(kwargs)
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)
    monkeypatch.setattr(
        "miniswewebagent.run.benchmarks.om2w.run_online_mind2web_judge",
        fake_run_online_mind2web_judge,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "gateway-key")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--workers",
            "1",
            "--evaluate",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
            "-c",
            "run.judge_endpoint=http://gateway.example/api/responses",
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 3
    assert all(row["endpoint_target_uri"] == "http://gateway.example/api/responses" for row in captured)
    assert all(row["api_key"] == "gateway-key" for row in captured)


def test_om2w_cli_allows_overriding_judge_runs(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "first",
                    "confirmed_task": "Open the first page.",
                    "website": "https://example.com/1",
                    "level": "hard",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_one(**kwargs):
        return {"final_response": "ok"}

    captured: list[dict[str, object]] = []

    def fake_run_online_mind2web_judge(**kwargs):
        captured.append(kwargs)
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)
    monkeypatch.setattr(
        "miniswewebagent.run.benchmarks.om2w.run_online_mind2web_judge",
        fake_run_online_mind2web_judge,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--workers",
            "1",
            "--evaluate",
            "--judge-runs",
            "1",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 1


def test_om2w_cli_supports_combined_task_levels_and_shards(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "easy-1",
                    "confirmed_task": "Open easy one.",
                    "website": "https://example.com/easy-1",
                    "level": "easy",
                },
                {
                    "task_id": "medium-1",
                    "confirmed_task": "Open medium one.",
                    "website": "https://example.com/medium-1",
                    "level": "medium",
                },
                {
                    "task_id": "hard-1",
                    "confirmed_task": "Open hard one.",
                    "website": "https://example.com/hard-1",
                    "level": "hard",
                },
                {
                    "task_id": "easy-2",
                    "confirmed_task": "Open easy two.",
                    "website": "https://example.com/easy-2",
                    "level": "easy",
                },
            ]
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)

    result = CliRunner().invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--task-level",
            "easy+hard",
            "--num-shards",
            "2",
            "--shard-index",
            "1",
            "--batch-name",
            "shared",
            "--workers",
            "1",
            "--no-evaluate",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
        ],
    )

    assert result.exit_code == 0
    assert [call["task_id"] for call in calls] == ["hard-1"]
    summary = json.loads((log_root / "shared" / "run_summary_shard1of2.json").read_text())
    assert summary["num_shards"] == 2
    assert summary["shard_index"] == 1
    assert summary["n_tasks"] == 1
    assert (log_root / "shared" / "generation_summary_shard1of2.json").exists()


def test_om2w_cli_resumes_completed_tasks(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "first",
                    "confirmed_task": "Open first.",
                    "website": "https://example.com/first",
                    "level": "easy",
                },
                {
                    "task_id": "second",
                    "confirmed_task": "Open second.",
                    "website": "https://example.com/second",
                    "level": "hard",
                },
            ]
        ),
        encoding="utf-8",
    )
    completed_dir = output_dir / "first"
    completed_dir.mkdir(parents=True)
    (completed_dir / "result.json").write_text('{"final_response": "done"}', encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)

    result = CliRunner().invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--task-level",
            "all",
            "--resume",
            "--batch-name",
            "resume",
            "--workers",
            "1",
            "--no-evaluate",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
        ],
    )

    assert result.exit_code == 0
    assert [call["task_id"] for call in calls] == ["second"]
    generation = json.loads((log_root / "resume" / "generation_summary.json").read_text())
    assert generation[0]["task_id"] == "first"
    assert generation[0]["status"] == "resumed"
    assert generation[1]["task_id"] == "second"
    assert generation[1]["status"] == "ok"
    summary = json.loads((log_root / "resume" / "run_summary.json").read_text())
    assert summary["n_tasks"] == 2
    assert summary["n_resumed"] == 1


def test_om2w_cli_retry_failed_replaces_failed_task_directory(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "failed",
                    "confirmed_task": "Retry failed.",
                    "website": "https://example.com/failed",
                    "level": "hard",
                }
            ]
        ),
        encoding="utf-8",
    )
    failed_dir = output_dir / "failed"
    failed_dir.mkdir(parents=True)
    (failed_dir / "result.json").write_text('{"run_exception": "timeout"}', encoding="utf-8")
    stale_file = failed_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)

    result = CliRunner().invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--resume",
            "--retry-failed",
            "--batch-name",
            "retry",
            "--workers",
            "1",
            "--no-evaluate",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
        ],
    )

    assert result.exit_code == 0
    assert [call["task_id"] for call in calls] == ["failed"]
    assert not stale_file.exists()


def test_om2w_cli_resume_replaces_unreadable_result(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "corrupt",
                    "confirmed_task": "Retry corrupt.",
                    "website": "https://example.com/corrupt",
                    "level": "hard",
                }
            ]
        ),
        encoding="utf-8",
    )
    corrupt_dir = output_dir / "corrupt"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "result.json").write_text("{", encoding="utf-8")
    stale_file = corrupt_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one_default", fake_run_one)

    result = CliRunner().invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--resume",
            "--batch-name",
            "corrupt",
            "--workers",
            "1",
            "--no-evaluate",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
        ],
    )

    assert result.exit_code == 0
    assert [call["task_id"] for call in calls] == ["corrupt"]
    assert not stale_file.exists()
    summary = json.loads((log_root / "corrupt" / "run_summary.json").read_text())
    assert summary["n_resumed"] == 0


def test_om2w_cli_judge_only_reports_missing_tasks_by_level(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "easy",
                    "confirmed_task": "Judge easy.",
                    "website": "https://example.com/easy",
                    "level": "easy",
                },
                {
                    "task_id": "hard",
                    "confirmed_task": "Judge hard.",
                    "website": "https://example.com/hard",
                    "level": "hard",
                },
            ]
        ),
        encoding="utf-8",
    )
    completed_dir = output_dir / "easy"
    completed_dir.mkdir(parents=True)
    (completed_dir / "result.json").write_text('{"final_response": "done"}', encoding="utf-8")

    def fail_if_generated(**kwargs):
        raise AssertionError(f"judge-only unexpectedly generated a task: {kwargs}")

    captured: list[dict[str, object]] = []

    def fake_run_online_mind2web_judge(**kwargs):
        captured.append(kwargs)
        eval_output_dir = kwargs["output_dir"]
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        result_file = (
            eval_output_dir
            / "WebJudge_Online_Mind2Web_Sandbox_eval_test-judge_score_threshold_3_auto_eval_results.json"
        )
        result_file.write_text(
            json.dumps({"task_id": "easy", "predicted_label": 1}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(
        "miniswewebagent.run.benchmarks.om2w.run_one_default",
        fail_if_generated,
    )
    monkeypatch.setattr(
        "miniswewebagent.run.benchmarks.om2w.run_online_mind2web_judge",
        fake_run_online_mind2web_judge,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    result = CliRunner().invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--task-level",
            "all",
            "--judge-only",
            "--judge-model",
            "test-judge",
            "--judge-runs",
            "1",
            "--judge-num-proc",
            "1",
            "--batch-name",
            "judge",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    summary = json.loads((log_root / "judge" / "run_summary_judge.json").read_text())
    assert summary["overall"] == {
        "success": 1,
        "total_tasks": 2,
        "success_rate": "50.0%",
    }
    assert summary["level_breakdown"]["easy"] == {
        "success": 1,
        "judged": 1,
        "total_tasks": 1,
    }
    assert summary["level_breakdown"]["hard"] == {
        "success": 0,
        "judged": 0,
        "total_tasks": 1,
    }


def test_om2w_cli_aggregates_judge_runs_by_task_majority(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    output_dir = tmp_path / "batch_output"
    log_root = tmp_path / "logs"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "hard",
                    "confirmed_task": "Judge hard.",
                    "website": "https://example.com/hard",
                    "level": "hard",
                }
            ]
        ),
        encoding="utf-8",
    )
    completed_dir = output_dir / "hard"
    completed_dir.mkdir(parents=True)
    (completed_dir / "result.json").write_text('{"final_response": "done"}', encoding="utf-8")

    def fake_run_online_mind2web_judge(**kwargs):
        eval_output_dir = kwargs["output_dir"]
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        predicted_label = 1 if eval_output_dir.name.endswith("_eval_1") else 0
        result_file = (
            eval_output_dir
            / "WebJudge_Online_Mind2Web_Sandbox_eval_test-judge_score_threshold_3_auto_eval_results.json"
        )
        result_file.write_text(
            json.dumps({"task_id": "hard", "predicted_label": predicted_label}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(
        "miniswewebagent.run.benchmarks.om2w.run_online_mind2web_judge",
        fake_run_online_mind2web_judge,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    result = CliRunner().invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--judge-only",
            "--judge-model",
            "test-judge",
            "--judge-runs",
            "3",
            "--judge-num-proc",
            "1",
            "--batch-name",
            "majority",
            "--output-dir",
            str(output_dir),
            "--log-root",
            str(log_root),
        ],
    )

    assert result.exit_code == 0
    summary = json.loads((log_root / "majority" / "run_summary_judge.json").read_text())
    assert summary["judge_aggregation"] == {
        "method": "per_task_majority_vote",
        "runs": 3,
    }
    assert summary["overall"] == {
        "success": 0,
        "total_tasks": 1,
        "success_rate": "0.0%",
    }
