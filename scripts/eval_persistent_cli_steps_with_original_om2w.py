#!/usr/bin/env python3
"""Evaluate persistent-browser trajectories using stored low-level actions.

This variant reuses ``eval_persistent_cli_with_original_om2w.py`` for task
discovery, judging, retries, and resumable output. It auto-detects either:

* task folders with ``task.json`` and ``steps/step_<id>.sh``; or
* task folders with ``result.json`` and ``screenshots/*.png``. For this layout,
  each ``action_history[].action`` is truncated after its final ``->``.

Both layouts read screenshots from ``screenshots/``, never from ``trajectory/``.
``trajectory/`` is a post-run export that pads failed captures with blank
placeholder PNGs so its indices stay dense; feeding those to the judge shows it
empty frames instead of real page state.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_persistent_cli_with_original_om2w as evaluator  # noqa: E402


STEP_FILE_RE = re.compile(r"^step_(\d+)\.sh$", re.IGNORECASE)
ACTION_HISTORY_SOURCE = "steps/step_<id>.sh"
ACTION_HISTORY_CONTRACT = (
    "full text of every non-empty steps/step_<id>.sh, ordered by numeric step ID"
)
RESULT_ACTION_HISTORY_SOURCE = "result.json.action_history[].action before final '->'"
RESULT_ACTION_HISTORY_CONTRACT = (
    "every result.json action_history[].action in list order, with the final "
    "'->' and all following text removed"
)
RAW_RESULT_ACTION_HISTORY_SOURCE = "result.json.action_history"
RAW_RESULT_ACTION_HISTORY_CONTRACT = (
    "every non-empty result.json action_history entry in list order, unchanged"
)
RESULT_SCREENSHOT_SOURCE = "screenshots/*.png"
RESULT_SCREENSHOT_CONTRACT = (
    "every root-level screenshots/*.png in browser-step order"
)


def load_step_action_history(browser_steps_path: Path) -> list[str]:
    """Load each ordered step script as one action-history entry."""
    steps_dir = browser_steps_path.parent / "steps"
    if not steps_dir.is_dir():
        return []

    step_files: list[tuple[int, str, Path]] = []
    for path in steps_dir.iterdir():
        match = STEP_FILE_RE.fullmatch(path.name)
        if path.is_file() and match:
            step_files.append((int(match.group(1)), path.name.lower(), path))
    step_files.sort()

    actions: list[str] = []
    for _, _, path in step_files:
        action = path.read_text(encoding="utf-8").strip()
        if action:
            actions.append(action)
    return actions


def trim_after_last_arrow(action: str) -> str:
    """Remove the final ``->`` delimiter and everything following it."""
    prefix, separator, _ = action.rpartition("->")
    return prefix.rstrip() if separator else action.strip()


def load_result_action_history(
    result_path: Path, *, trim_last_arrow: bool
) -> list[str]:
    """Load ordered string or object actions from a result.json artifact."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    history = payload.get("action_history")
    if not isinstance(history, list):
        raise ValueError(f"{result_path}: action_history must be a list")

    actions: list[str] = []
    for index, row in enumerate(history):
        if isinstance(row, str):
            action = row
        elif isinstance(row, dict) and isinstance(row.get("action"), str):
            action = row["action"]
        else:
            raise ValueError(
                f"{result_path}: action_history[{index}] must be a string "
                "or an object with a string .action field"
            )
        if trim_last_arrow:
            action = trim_after_last_arrow(action)
        if action.strip():
            actions.append(action)
    return actions


def load_result_task_artifacts(
    task_dir: Path, *, trim_last_arrow: bool
) -> evaluator.TaskArtifacts:
    result_path = task_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    task_id = str(payload.get("task_id") or task_dir.name).strip()
    task = str(payload.get("task") or "").strip()
    if not task:
        raise ValueError(f"{result_path}: missing task description")

    return evaluator.TaskArtifacts(
        task_id=task_id,
        task_dir=str(task_dir.resolve()),
        task=task,
        action_history=load_result_action_history(
            result_path, trim_last_arrow=trim_last_arrow
        ),
        screenshot_paths=evaluator.load_screenshot_paths(task_dir / "screenshots"),
    )


def discover_result_task_artifacts(
    trajectories_dir: Path,
    *,
    trim_last_arrow: bool,
) -> list[evaluator.TaskArtifacts]:
    task_dirs = sorted(
        path
        for path in trajectories_dir.iterdir()
        if path.is_dir() and (path / "result.json").is_file()
    )
    artifacts = [
        load_result_task_artifacts(path, trim_last_arrow=trim_last_arrow)
        for path in task_dirs
    ]
    task_ids = [item.task_id for item in artifacts]
    if len(task_ids) != len(set(task_ids)):
        duplicates = sorted(
            task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1
        )
        raise ValueError(f"Duplicate task IDs: {duplicates}")
    # A missing screenshots/ directory is a layout mismatch, not a bad task:
    # judging would silently fall back to action text alone for the whole run.
    # Individual tasks with no captures stay tolerated.
    if artifacts and not any(item.screenshot_paths for item in artifacts):
        raise ValueError(
            f"{trajectories_dir}: no task directory contains root-level "
            "screenshots/*.png, so every task would be judged without images"
        )
    return artifacts


def uses_result_json_layout(trajectories_dir: Path) -> bool:
    task_dirs = [path for path in trajectories_dir.iterdir() if path.is_dir()]
    has_task_json = any((path / "task.json").is_file() for path in task_dirs)
    has_result_json = any((path / "result.json").is_file() for path in task_dirs)
    if has_task_json:
        return False
    if has_result_json:
        return True
    raise ValueError(
        f"{trajectories_dir}: no task folders containing task.json or result.json"
    )


def main() -> None:
    args = evaluator.parse_args(
        description=(
            "Run upstream WebJudge_Online_Mind2Web_eval over persistent-browser "
            "step scripts or result.json low-level actions."
        )
    )
    trajectories_dir = Path(args.trajectories_dir).resolve()
    result_mode = args.result_action_history_mode
    if result_mode is not None or uses_result_json_layout(trajectories_dir):
        trim_last_arrow = result_mode != "raw"
        args.task_source = "result.json.task"
        if trim_last_arrow:
            args.action_history_source = RESULT_ACTION_HISTORY_SOURCE
            args.action_history_contract = RESULT_ACTION_HISTORY_CONTRACT
        else:
            args.action_history_source = RAW_RESULT_ACTION_HISTORY_SOURCE
            args.action_history_contract = RAW_RESULT_ACTION_HISTORY_CONTRACT
        args.screenshot_source = RESULT_SCREENSHOT_SOURCE
        args.screenshot_contract = RESULT_SCREENSHOT_CONTRACT
        evaluator.discover_task_artifacts = lambda root: discover_result_task_artifacts(
            root,
            trim_last_arrow=trim_last_arrow,
        )
    else:
        args.action_history_source = ACTION_HISTORY_SOURCE
        args.action_history_contract = ACTION_HISTORY_CONTRACT
        evaluator.load_action_history = load_step_action_history
    evaluator.parallel_eval(args)


if __name__ == "__main__":
    main()
