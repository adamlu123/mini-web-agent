from __future__ import annotations

import json

from typer.testing import CliRunner

from miniswewebagent.run.benchmarks.om2w import DEFAULT_OM2W_CONFIGS, app


def test_om2w_cli_runs_selected_tasks(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "first",
                    "confirmed_task": "Open the first page.",
                    "website": "https://example.com/1",
                },
                {
                    "task_id": "second",
                    "confirmed_task": "Open the second page.",
                    "website": "https://example.com/2",
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
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["task_id"] == "second"
    assert calls[0]["task"] == "Open the second page."
    assert calls[0]["start_url"] == "https://example.com/2"
    assert calls[0]["config_spec"] == DEFAULT_OM2W_CONFIGS


def test_om2w_cli_respects_limit(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "om2w.json"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "first",
                    "confirmed_task": "Open the first page.",
                    "website": "https://example.com/1",
                },
                {
                    "task_id": "second",
                    "confirmed_task": "Open the second page.",
                    "website": "https://example.com/2",
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
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["task_id"] == "first"
    assert calls[0]["config_spec"] == DEFAULT_OM2W_CONFIGS