from __future__ import annotations

import json

from miniswewebagent.utils.om2w_tasks import load_om2w_task, load_om2w_tasks


def test_load_om2w_tasks_reads_online_mind2web_json(tmp_path) -> None:
    tasks_file = tmp_path / "om2w.json"
    tasks_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "t1",
                    "confirmed_task": "  Find the cheapest flight.  ",
                    "website": " https://example.com/ ",
                    "level": "medium",
                    "reference_length": 6,
                }
            ]
        )
    )

    (task,) = load_om2w_tasks(tasks_file)

    assert task["task_id"] == "t1"
    assert task["task"] == "Find the cheapest flight."
    assert task["start_url"] == "https://example.com/"
    assert task["level"] == "medium"
    assert task["reference_length"] == 6


def test_load_om2w_tasks_reads_fara_style_csv(tmp_path) -> None:
    tasks_file = tmp_path / "fara.csv"
    tasks_file.write_text(
        "task_id,instruction,start_page,level,estimated_steps\n"
        'm2w_exp_1,"Find a sushi roll recipe, then play its video.",'
        "https://www.allrecipes.com/,hard,9\n",
        encoding="utf-8",
    )

    (task,) = load_om2w_tasks(tasks_file)

    assert task["task_id"] == "m2w_exp_1"
    assert task["task"] == "Find a sushi roll recipe, then play its video."
    assert task["start_url"] == "https://www.allrecipes.com/"
    assert task["level"] == "hard"
    assert task["reference_length"] == 9


def test_load_om2w_task_looks_up_a_csv_row_by_id(tmp_path) -> None:
    tasks_file = tmp_path / "fara.csv"
    tasks_file.write_text(
        "task_id,instruction,start_page,level,estimated_steps\n"
        "a,First task,https://a.example/,hard,3\n"
        "b,Second task,https://b.example/,hard,4\n",
        encoding="utf-8",
    )

    assert load_om2w_task(tasks_file, "b")["task"] == "Second task"


def test_load_om2w_tasks_tolerates_non_numeric_reference_length(tmp_path) -> None:
    tasks_file = tmp_path / "fara.csv"
    tasks_file.write_text(
        "task_id,instruction,start_page,estimated_steps\n"
        "a,First task,https://a.example/,unknown\n",
        encoding="utf-8",
    )

    assert load_om2w_tasks(tasks_file)[0]["reference_length"] == 0
