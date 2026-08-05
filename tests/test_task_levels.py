from __future__ import annotations

import json

from miniswewebagent.run.utilities.task_levels import load_task_levels


def test_load_task_levels_reads_bundled_om2w_and_odysseys_files() -> None:
    levels = load_task_levels()

    # Known sample rows, cross-checked against the bundled JSON files.
    assert levels["b7258ee05d75e6c50673a59914db412e_110325"] == "medium"
    assert levels["440ed7f388a2a4528a8d9fb75f83e11f934b5b5d"] == "easy"

    assert set(levels.values()) <= {"easy", "medium", "hard", ""}
    # 300 om2w tasks + 200 odysseys tasks, all with distinct ids in practice.
    assert len(levels) >= 490


def test_load_task_levels_skips_missing_sources(tmp_path) -> None:
    missing_file = tmp_path / "does-not-exist.json"

    levels = load_task_levels(sources=(str(missing_file),))

    assert levels == {}


def test_load_task_levels_merges_custom_sources_and_normalizes_level(tmp_path) -> None:
    first_file = tmp_path / "first.json"
    first_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "custom-1",
                    "confirmed_task": "Do a thing.",
                    "website": "https://example.com",
                    "level": "Hard",
                }
            ]
        ),
        encoding="utf-8",
    )
    second_file = tmp_path / "second.json"
    second_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "custom-1",
                    "confirmed_task": "Do a thing, overridden.",
                    "website": "https://example.com",
                    "level": "easy",
                },
                {
                    "task_id": "custom-2",
                    "confirmed_task": "Do another thing.",
                    "website": "https://example.com",
                    "level": "medium",
                },
            ]
        ),
        encoding="utf-8",
    )

    levels = load_task_levels(sources=(str(first_file), str(second_file)))

    # Later sources win on task_id collisions, and level is lowercased.
    assert levels == {"custom-1": "easy", "custom-2": "medium"}
