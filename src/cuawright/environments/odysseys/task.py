from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ODYSSEYS_DEFAULT_DOMAIN = "mind2web_chrome"
_TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class OdysseysTask:
    task_id: str
    instruction: str
    website: str
    level: str
    reference_length: int
    rubrics: Mapping[str, Mapping[str, str]]
    task_record: Mapping[str, Any]
    example: Mapping[str, Any]
    dataset_path: Path
    example_path: Path
    dataset_bytes: bytes
    example_bytes: bytes
    dataset_sha256: str
    example_sha256: str


def load_odysseys_task(
    *,
    task_id: str,
    dataset_path: str | Path,
    examples_dir: str | Path,
    domain: str = ODYSSEYS_DEFAULT_DOMAIN,
) -> OdysseysTask:
    normalized_id = _task_id(task_id)
    dataset = Path(dataset_path).expanduser().resolve()
    examples = Path(examples_dir).expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"Odysseys task source not found: {dataset}")
    if not domain or Path(domain).name != domain:
        raise ValueError("Odysseys domain must be one safe path component")

    dataset_bytes = dataset.read_bytes()
    try:
        raw_tasks = json.loads(dataset_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Odysseys task source is not valid JSON: {dataset}") from exc
    if not isinstance(raw_tasks, list):
        raise TypeError("Odysseys task source must contain a JSON list")
    matches = [
        item
        for item in raw_tasks
        if isinstance(item, Mapping)
        and str(item.get("task_id", "")).lower() == normalized_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Odysseys task {normalized_id}, found {len(matches)}"
        )
    metadata = matches[0]
    instruction = _required_text(metadata, "confirmed_task")
    website = _required_text(metadata, "website")
    level = _required_text(metadata, "level")
    reference_length = metadata.get("reference_length")
    if (
        isinstance(reference_length, bool)
        or not isinstance(reference_length, int)
        or reference_length < 1
    ):
        raise ValueError("Odysseys reference_length must be a positive integer")
    rubrics = _rubrics(metadata.get("rubrics"))

    example_path = examples / "examples" / domain / f"{normalized_id}.json"
    if not example_path.is_file():
        raise FileNotFoundError(f"Odysseys OSWorld example not found: {example_path}")
    example_bytes = example_path.read_bytes()
    try:
        example = json.loads(example_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Odysseys example is not valid JSON: {example_path}") from exc
    if not isinstance(example, Mapping):
        raise TypeError("Odysseys OSWorld example must contain a JSON object")
    if str(example.get("id", "")).lower() != normalized_id:
        raise ValueError("Odysseys example id does not match the selected task")
    if example.get("instruction") != instruction:
        raise ValueError("Odysseys example instruction differs from the task source")
    if example.get("source") != website:
        raise ValueError("Odysseys example start website differs from the task source")
    if not isinstance(example.get("config"), list):
        raise TypeError("Odysseys example config must be a list")

    return OdysseysTask(
        task_id=normalized_id,
        instruction=instruction,
        website=website,
        level=level,
        reference_length=reference_length,
        rubrics=rubrics,
        task_record=dict(metadata),
        example=dict(example),
        dataset_path=dataset,
        example_path=example_path,
        dataset_bytes=dataset_bytes,
        example_bytes=example_bytes,
        dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
        example_sha256=hashlib.sha256(example_bytes).hexdigest(),
    )


def task_ids(dataset_path: str | Path) -> tuple[str, ...]:
    path = Path(dataset_path).expanduser().resolve()
    value = json.loads(path.read_bytes())
    if not isinstance(value, list):
        raise TypeError("Odysseys task source must contain a JSON list")
    result = tuple(
        _task_id(str(item.get("task_id", "")))
        for item in value
        if isinstance(item, Mapping)
    )
    if len(result) != len(value):
        raise ValueError("Every Odysseys dataset row must contain a valid task_id")
    if len(result) != len(set(result)):
        raise ValueError("Odysseys task ids must be unique")
    return result


def _task_id(value: str) -> str:
    if not isinstance(value, str) or not _TASK_ID_PATTERN.fullmatch(value.lower()):
        raise ValueError("Odysseys task_id must be a 40-character hexadecimal SHA-1")
    return value.lower()


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Odysseys {key} must be non-empty text")
    return item


def _rubrics(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Odysseys rubrics must be a non-empty object")
    result: dict[str, dict[str, str]] = {}
    for rubric_id, rubric in value.items():
        if not isinstance(rubric_id, str) or not rubric_id.strip():
            raise ValueError("Odysseys rubric ids must be non-empty strings")
        if not isinstance(rubric, Mapping):
            raise TypeError(f"Odysseys rubric {rubric_id} must be an object")
        result[rubric_id] = {
            "requirement": _required_text(rubric, "requirement"),
            "verification": _required_text(rubric, "verification"),
        }
    return result
