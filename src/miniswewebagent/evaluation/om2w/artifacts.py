"""Evaluate persistent-browser trajectories using stored low-level actions.

This variant reuses the package evaluator for task
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
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from miniswewebagent.evaluation.om2w import runner as evaluator

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


class ArtifactLayout(str, Enum):
    """Supported on-disk trajectory layouts for the steps evaluator."""

    AUTO = "auto"
    STEP_SCRIPTS = "step-scripts"
    RESULT_JSON = "result-json"


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
        raise ValueError(  # noqa: TRY004 - malformed artifact content
            f"{result_path}: action_history must be a list"
        )

    actions: list[str] = []
    for index, row in enumerate(history):
        if isinstance(row, str):
            action = row
        elif isinstance(row, dict) and isinstance(row.get("action"), str):
            action = row["action"]
        else:
            raise ValueError(  # noqa: TRY004 - malformed artifact content
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


def resolve_artifact_loader(
    trajectories_dir: Path,
    *,
    layout: ArtifactLayout | str = ArtifactLayout.AUTO,
    result_action_history_mode: str | None = None,
) -> tuple[Callable[[Path], list[evaluator.TaskArtifacts]], dict[str, str]]:
    """Resolve an explicit artifact layout while retaining legacy auto-detection."""
    layout = ArtifactLayout(layout)
    if layout is ArtifactLayout.AUTO:
        layout = (
            ArtifactLayout.RESULT_JSON
            if result_action_history_mode is not None
            or uses_result_json_layout(trajectories_dir)
            else ArtifactLayout.STEP_SCRIPTS
        )

    if layout is ArtifactLayout.RESULT_JSON:
        trim_last_arrow = result_action_history_mode != "raw"
        if trim_last_arrow:
            action_source = RESULT_ACTION_HISTORY_SOURCE
            action_contract = RESULT_ACTION_HISTORY_CONTRACT
        else:
            action_source = RAW_RESULT_ACTION_HISTORY_SOURCE
            action_contract = RAW_RESULT_ACTION_HISTORY_CONTRACT
        return (
            lambda root: discover_result_task_artifacts(
                root,
                trim_last_arrow=trim_last_arrow,
            ),
            {
                "task_source": "result.json.task",
                "action_history_source": action_source,
                "action_history_contract": action_contract,
                "screenshot_source": RESULT_SCREENSHOT_SOURCE,
                "screenshot_contract": RESULT_SCREENSHOT_CONTRACT,
            },
        )

    def discover_step_artifacts(root: Path) -> list[evaluator.TaskArtifacts]:
        return evaluator.discover_task_artifacts(
            root,
            action_history_loader=load_step_action_history,
        )

    return (
        discover_step_artifacts,
        {
            "task_source": evaluator.DEFAULT_TASK_SOURCE,
            "action_history_source": ACTION_HISTORY_SOURCE,
            "action_history_contract": ACTION_HISTORY_CONTRACT,
            "screenshot_source": evaluator.DEFAULT_SCREENSHOT_SOURCE,
            "screenshot_contract": evaluator.DEFAULT_SCREENSHOT_CONTRACT,
        },
    )


def main() -> None:
    args = evaluator.parse_args(
        description=(
            "Run upstream WebJudge_Online_Mind2Web_eval over persistent-browser "
            "step scripts or result.json low-level actions."
        ),
        include_artifact_layout=True,
    )
    trajectories_dir = Path(args.trajectories_dir).resolve()
    artifact_loader, contracts = resolve_artifact_loader(
        trajectories_dir,
        layout=args.artifact_layout,
        result_action_history_mode=args.result_action_history_mode,
    )
    for name, value in contracts.items():
        setattr(args, name, value)
    evaluator.parallel_eval(args, artifact_loader=artifact_loader)


if __name__ == "__main__":
    main()
