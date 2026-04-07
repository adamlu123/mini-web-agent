from __future__ import annotations

import json

from typer.testing import CliRunner

from miniswewebagent.run.benchmarks.webtail_flight_hotel import DEFAULT_WEBTAIL_CONFIGS, app, load_webtail_flight_hotel_tasks


def test_webtail_cli_defaults_run_without_showing_help() -> None:
    assert app.info.no_args_is_help is False


def test_load_webtail_tasks_supports_metadata_wrapper(tmp_path) -> None:
    tasks_file = tmp_path / "webtail.json"
    tasks_file.write_text(
        json.dumps(
            {
                "metadata": {"total_tasks": 2},
                "tasks": [
                    {
                        "task_id": "flight__one",
                        "task": "Search for flight one.",
                        "start_url": "https://example.com/flight",
                        "source_family": "flight",
                    },
                    {
                        "task_id": "hotel__one",
                        "task": "Search for hotel one.",
                        "website_url": "https://example.com/hotel",
                        "source_family": "hotel",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_webtail_flight_hotel_tasks(tasks_file)

    assert [task["task_id"] for task in tasks] == ["flight__one", "hotel__one"]
    assert tasks[0]["start_url"] == "https://example.com/flight"
    assert tasks[1]["start_url"] == "https://example.com/hotel"


def test_webtail_cli_filters_and_runs_selected_tasks(tmp_path, monkeypatch) -> None:
    tasks_file = tmp_path / "webtail.json"
    output_dir = tmp_path / "batch_output"
    tasks_file.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "flight__one",
                        "task": "Search for flight one.",
                        "start_url": "https://example.com/flight",
                        "source_family": "flight",
                    },
                    {
                        "task_id": "hotel__two",
                        "task": "Search for hotel two.",
                        "start_url": "https://example.com/hotel",
                        "source_family": "hotel",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_run_one(**kwargs):
        calls.append(kwargs)
        return {"final_response": "ok"}

    monkeypatch.setattr("miniswewebagent.run.benchmarks.webtail_flight_hotel.run_one", fake_run_one)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--tasks-file",
            str(tasks_file),
            "--family",
            "hotel",
            "--workers",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["task_id"] == "hotel__two"
    assert calls[0]["task"] == "Search for hotel two."
    assert calls[0]["start_url"] == "https://example.com/hotel"
    assert calls[0]["config_spec"] == DEFAULT_WEBTAIL_CONFIGS
    assert calls[0]["resolved_output_dir"].name == "hotel__two"
    assert (output_dir / "config_snapshot" / "config_spec_manifest.json").exists()
    assert (output_dir / "config_snapshot" / "merged_config.yaml").exists()
