from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_om2w_tasks(tasks_file: str | Path) -> list[dict[str, Any]]:
    path = Path(tasks_file).expanduser()
    payload = json.loads(path.read_text())
    return [normalize_om2w_task(item) for item in payload]


def load_om2w_task(tasks_file: str | Path, task_id: str) -> dict[str, Any]:
    for item in load_om2w_tasks(tasks_file):
        if item["task_id"] == task_id:
            return item
    raise ValueError(f"Task id not found in {tasks_file}: {task_id}")


def normalize_om2w_task(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(item.get("task_id", "")),
        "task": str(item.get("confirmed_task", "")).strip(),
        "start_url": str(item.get("website", "")).strip(),
        "level": str(item.get("level", "")).strip(),
        "reference_length": int(item.get("reference_length", 0) or 0),
        "raw": item,
    }
