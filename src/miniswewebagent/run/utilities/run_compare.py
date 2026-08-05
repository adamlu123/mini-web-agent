"""Compare Online-Mind2Web-style benchmark runs living under a viewer root.

Given two or more run directories (each shaped like an ``outputs/<run_id>``
batch produced by ``miniswewebagent.run.benchmarks.om2w``), this module
resolves a per-task success/failure status from the judge's
``WebJudge_*_auto_eval_results.json`` files, then produces:

* a per-run leaderboard (overall + per-level success rate), and
* a task-by-task diff against a chosen baseline run, classifying every task
  as improved, regressed, unchanged, or unknown.

This module is intentionally self-contained (it does not import from
``trace_viewer.py``) so it can be developed, tested, and reviewed
independently of the HTTP/CLI layer that calls it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Per-task status for a single run.
STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"
STATUS_UNKNOWN = "unknown"
STATUS_MISSING = "missing"

# Task-level classification of a candidate run against the baseline run.
FLIP_IMPROVED = "improved"
FLIP_REGRESSED = "regressed"
FLIP_SAME_SUCCESS = "same_success"
FLIP_SAME_FAIL = "same_fail"
FLIP_UNKNOWN = "unknown"

# Bucket key used for tasks whose level could not be resolved.
UNKNOWN_LEVEL = "unknown"

_JUDGE_GLOB = "WebJudge_*_auto_eval_results.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Best-effort JSONL reader: skips a file entirely if it cannot be parsed.

    Compare runs are often read while a batch is still in flight, so a
    partially-written judge file should not take down the whole comparison.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _reduce_status(labels: Sequence[Any]) -> str:
    """Reduce every ``predicted_label`` seen for one task to one status.

    A task may be scored by several parallel judge runs (``judge_runs`` in
    ``om2w.py``, typically 3). We take a majority vote over the informative
    labels (``1`` success / ``0`` failure) and fall back to unknown when
    there is no informative label at all, or a tie.
    """
    success_votes = sum(1 for label in labels if label == 1)
    failure_votes = sum(1 for label in labels if label == 0)
    if success_votes == 0 and failure_votes == 0:
        return STATUS_UNKNOWN
    if success_votes > failure_votes:
        return STATUS_SUCCESS
    if failure_votes > success_votes:
        return STATUS_FAILURE
    return STATUS_UNKNOWN


def load_run_task_statuses(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Resolve ``{task_id: {"title", "status", "folderName"}}`` for one run dir.

    Tasks are discovered from ``<run_dir>/<task>/result.json``. Status comes
    from majority vote over any ``WebJudge_*_auto_eval_results.json`` rows
    matching that task id; a task with no matching judge row is
    ``STATUS_UNKNOWN`` (present, but not yet judged), not ``STATUS_MISSING``
    (``STATUS_MISSING`` is reserved for tasks absent from this run dir
    entirely, and is only ever assigned by ``compare_runs``).

    ``folderName`` is the task's directory name (``task_dir.name``), which is
    what the existing trace-viewer ``/api/task?task=`` endpoint expects. It
    usually equals ``task_id`` for om2w batch runs, but callers that need to
    link into a specific task's detail view should use ``folderName``, not
    ``task_id``, since the two are not guaranteed to match for every run
    layout.
    """
    if not run_dir.is_dir():
        return {}

    entries: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        result_path = task_dir / "result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        task_id = str(result.get("task_id") or task_dir.name).strip() or task_dir.name
        entries[task_id] = {
            "title": str(result.get("task") or ""),
            "folderName": task_dir.name,
            "labels": [],
        }

    for judge_path in sorted(run_dir.glob(_JUDGE_GLOB)):
        for row in _load_jsonl(judge_path):
            task_id = str(row.get("task_id") or "").strip()
            if not task_id:
                continue
            # A judge row with no matching result.json (unexpected, but not
            # fatal) falls back to the task id as its own folder name guess.
            entry = entries.setdefault(task_id, {"title": "", "folderName": task_id, "labels": []})
            entry["labels"].append(row.get("predicted_label"))

    return {
        task_id: {
            "title": entry["title"],
            "status": _reduce_status(entry["labels"]),
            "folderName": entry["folderName"],
        }
        for task_id, entry in entries.items()
    }


def _empty_bucket() -> dict[str, int]:
    return {"success": 0, "failure": 0, "unknown": 0, "total": 0}


def _bucket_with_rate(bucket: dict[str, int]) -> dict[str, Any]:
    total = bucket["total"]
    success_rate = (bucket["success"] / total) if total else 0.0
    return {**bucket, "successRate": success_rate}


def summarize_run(
    run_id: str,
    statuses: Mapping[str, Mapping[str, Any]],
    task_levels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one leaderboard row: overall + per-level success-rate counts."""
    task_levels = task_levels or {}
    overall = _empty_bucket()
    by_level: dict[str, dict[str, int]] = {}

    for task_id, entry in statuses.items():
        status = entry.get("status", STATUS_UNKNOWN)
        level = task_levels.get(task_id) or UNKNOWN_LEVEL
        bucket = by_level.setdefault(level, _empty_bucket())
        for target in (overall, bucket):
            target["total"] += 1
            if status in (STATUS_SUCCESS, STATUS_FAILURE, STATUS_UNKNOWN):
                target[status] += 1

    return {
        "runId": run_id,
        "totalTasks": len(statuses),
        "overall": _bucket_with_rate(overall),
        "byLevel": {level: _bucket_with_rate(bucket) for level, bucket in sorted(by_level.items())},
    }


def classify_flip(baseline_status: str, candidate_status: str) -> str:
    """Classify one task's candidate-run outcome relative to the baseline run."""
    if baseline_status == STATUS_SUCCESS and candidate_status == STATUS_FAILURE:
        return FLIP_REGRESSED
    if baseline_status == STATUS_FAILURE and candidate_status == STATUS_SUCCESS:
        return FLIP_IMPROVED
    if baseline_status == STATUS_SUCCESS and candidate_status == STATUS_SUCCESS:
        return FLIP_SAME_SUCCESS
    if baseline_status == STATUS_FAILURE and candidate_status == STATUS_FAILURE:
        return FLIP_SAME_FAIL
    return FLIP_UNKNOWN


def _new_diff_counter() -> dict[str, Any]:
    return {
        "improved": 0,
        "regressed": 0,
        "sameSuccess": 0,
        "sameFail": 0,
        "unknown": 0,
        "byLevel": {},
    }


_FLIP_TO_COUNTER_KEY = {
    FLIP_IMPROVED: "improved",
    FLIP_REGRESSED: "regressed",
    FLIP_SAME_SUCCESS: "sameSuccess",
    FLIP_SAME_FAIL: "sameFail",
    FLIP_UNKNOWN: "unknown",
}


def _accumulate_diff(counter: dict[str, Any], flip: str, level: str) -> None:
    key = _FLIP_TO_COUNTER_KEY[flip]
    counter[key] += 1
    level_bucket = counter["byLevel"].setdefault(
        level, {"improved": 0, "regressed": 0, "sameSuccess": 0, "sameFail": 0, "unknown": 0}
    )
    level_bucket[key] += 1


def compare_runs(
    root_dir: Path,
    run_ids: Sequence[str],
    *,
    baseline_id: str | None = None,
    task_levels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare 1+ runs under ``root_dir``, diffing each against a baseline.

    Args:
        root_dir: Directory containing one subdirectory per run id (the same
            root passed to ``mini-web-traces --root``).
        run_ids: Run directory names to compare, e.g. ``["baseline", "candidate"]``.
        baseline_id: Which of ``run_ids`` is the baseline for flip
            classification. Defaults to ``run_ids[0]``.
        task_levels: Optional ``{task_id: level}`` lookup (e.g. sourced from
            the bundled Online-Mind2Web / Odyssey task JSON files) used to
            break leaderboard and diff counts down by difficulty level. Tasks
            missing from this mapping are bucketed under ``UNKNOWN_LEVEL``.

    Returns:
        A JSON-serializable dict with ``leaderboard`` (one row per run id),
        ``tasks`` (per-task statuses across all runs, a ``runDirNames``
        map of ``{run_id: folder_name}`` for linking into the existing
        ``/api/task`` detail endpoint, and ``flipsVsBaseline``), and
        ``diffSummary`` (aggregate flip counts per candidate run, overall and
        by level).
    """
    if not run_ids:
        raise ValueError("compare_runs requires at least one run_id")

    run_ids = list(run_ids)
    resolved_baseline = baseline_id or run_ids[0]
    if resolved_baseline not in run_ids:
        raise ValueError(f"baseline_id {resolved_baseline!r} must be one of run_ids {run_ids!r}")

    task_levels = task_levels or {}
    root_dir = Path(root_dir)

    per_run_statuses = {run_id: load_run_task_statuses(root_dir / run_id) for run_id in run_ids}
    leaderboard = [summarize_run(run_id, per_run_statuses[run_id], task_levels) for run_id in run_ids]

    all_task_ids: set[str] = set()
    for statuses in per_run_statuses.values():
        all_task_ids.update(statuses)

    diff_counters = {run_id: _new_diff_counter() for run_id in run_ids if run_id != resolved_baseline}

    tasks: list[dict[str, Any]] = []
    for task_id in sorted(all_task_ids):
        level = task_levels.get(task_id) or UNKNOWN_LEVEL
        title = _first_title(per_run_statuses, run_ids, task_id)
        statuses_row = {
            run_id: per_run_statuses[run_id].get(task_id, {}).get("status", STATUS_MISSING)
            for run_id in run_ids
        }
        run_dir_names = {
            run_id: per_run_statuses[run_id][task_id]["folderName"]
            for run_id in run_ids
            if task_id in per_run_statuses[run_id]
        }
        baseline_status = statuses_row[resolved_baseline]

        flips: dict[str, str] = {}
        for run_id in run_ids:
            if run_id == resolved_baseline:
                continue
            flip = classify_flip(baseline_status, statuses_row[run_id])
            flips[run_id] = flip
            _accumulate_diff(diff_counters[run_id], flip, level)

        tasks.append(
            {
                "taskId": task_id,
                "level": level,
                "title": title,
                "statuses": statuses_row,
                "runDirNames": run_dir_names,
                "flipsVsBaseline": flips,
            }
        )

    return {
        "rootDir": str(root_dir),
        "runIds": run_ids,
        "baselineId": resolved_baseline,
        "leaderboard": leaderboard,
        "tasks": tasks,
        "diffSummary": diff_counters,
    }


def _first_title(
    per_run_statuses: Mapping[str, Mapping[str, Mapping[str, Any]]],
    run_ids: Iterable[str],
    task_id: str,
) -> str:
    for run_id in run_ids:
        title = per_run_statuses[run_id].get(task_id, {}).get("title", "")
        if title:
            return str(title)
    return ""
