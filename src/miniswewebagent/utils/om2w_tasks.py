from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Column/field aliases so Online-Mind2Web JSON dumps and the Fara-style task CSVs
# (task_id,instruction,start_page,...) load through the same normalizer.
_TASK_KEYS = ("confirmed_task", "instruction", "task")
_START_URL_KEYS = ("website", "start_page", "start_url")
_REFERENCE_LENGTH_KEYS = ("reference_length", "estimated_steps")


def load_om2w_tasks(tasks_file: str | Path) -> list[dict[str, Any]]:
    path = Path(tasks_file).expanduser()
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            payload: list[dict[str, Any]] = list(csv.DictReader(handle))
    else:
        payload = json.loads(path.read_text())
    return [normalize_om2w_task(item) for item in payload]


def load_om2w_task(tasks_file: str | Path, task_id: str) -> dict[str, Any]:
    for item in load_om2w_tasks(tasks_file):
        if item["task_id"] == task_id:
            return item
    raise ValueError(f"Task id not found in {tasks_file}: {task_id}")


def _first_value(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_om2w_task(item: dict[str, Any]) -> dict[str, Any]:
    try:
        reference_length = int(float(_first_value(item, _REFERENCE_LENGTH_KEYS) or 0))
    except ValueError:
        reference_length = 0
    return {
        "task_id": str(item.get("task_id", "")),
        "task": _first_value(item, _TASK_KEYS),
        "start_url": _first_value(item, _START_URL_KEYS),
        "level": str(item.get("level", "")).strip(),
        "reference_length": reference_length,
        "raw": item,
    }
