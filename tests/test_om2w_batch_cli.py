from __future__ import annotations

import json

from typer.testing import CliRunner

from miniswewebagent.run.benchmarks.om2w import DEFAULT_OM2W_CONFIGS, app


def test_om2w_cli_defaults_run_without_showing_help() -> None:
    assert app.info.no_args_is_help is False


def test_om2w_cli_runs_selected_tasks(tmp_path, monkeypatch) -> None:
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

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one", fake_run_one)

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
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["task_id"] == "second"
    assert calls[0]["task"] == "Open the second page."
    assert calls[0]["start_url"] == "https://example.com/2"
    assert calls[0]["config_spec"] == DEFAULT_OM2W_CONFIGS
    assert calls[0]["resolved_output_dir"].name == "second"


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

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one", fake_run_one)

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

    monkeypatch.setattr("miniswewebagent.run.benchmarks.om2w.run_one", fake_run_one)

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
