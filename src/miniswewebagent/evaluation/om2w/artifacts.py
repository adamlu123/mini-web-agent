"""Read Online-Mind2Web trajectories off disk, in any of the supported layouts.

This module owns everything between the filesystem and the judge: the
:class:`TaskArtifacts` record, generic task discovery, the on-disk layouts
enumerated by :class:`ArtifactLayout`, and the :class:`ArtifactSpec` that binds a
layout to the provenance strings describing it.
:mod:`miniswewebagent.evaluation.om2w.runner` consumes a spec; it never reaches
into a trajectory directory itself.

A layout only decides the **action history** — what the judge is told the agent
did. The task text and the screenshots are read the same way regardless, so two
layouts over one directory produce two different scores from the same run:

* ``browser-steps`` — ``task.json`` + each ``browser-steps.jsonl`` row's
  natural-language ``action``. What a persistent-CLI run
  (``generation/best_default_judge_json_persistent_cli.yaml``) describes itself
  as doing, and what ``scripts/eval/persistent_cli.py`` scores.
* ``step-scripts`` — ``task.json`` + the full text of each
  ``steps/step_<id>.sh``, i.e. the shell commands actually executed. The default
  for ``scripts/eval/persistent_cli_steps.py``, and therefore for a batch run
  through ``run/benchmarks/om2w.py`` that sets no ``run.judge_script``.
* ``result-json`` — ``result.json``'s ``action_history``, for older runs that
  never wrote ``task.json``.

A persistent-CLI run writes the artifacts for the first two layouts on every
task, so which one a scored run used is a choice made by the caller, never a
property of the directory. It is recorded in ``eval_manifest.json`` and on every
result row; ``ArtifactLayout.AUTO`` cannot recover it.

Every layout reads screenshots from ``screenshots/``, never from ``trajectory/``.
``trajectory/`` is a post-run export that pads failed captures with blank
placeholder PNGs so its indices stay dense; feeding those to the judge shows it
empty frames instead of real page state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Literal

DEFAULT_TASK_SOURCE = "task.json.task"
DEFAULT_ACTION_HISTORY_SOURCE = "browser-steps.jsonl.action"
DEFAULT_ACTION_HISTORY_CONTRACT = "every non-empty browser-steps.jsonl.action in file order"
DEFAULT_SCREENSHOT_SOURCE = "screenshots/*.png"
DEFAULT_SCREENSHOT_CONTRACT = "every root-level screenshots/*.png in browser-step order"

STEP_FILE_RE = re.compile(r"^step_(\d+)\.sh$", re.IGNORECASE)
STEP_ACTION_HISTORY_SOURCE = "steps/step_<id>.sh"
STEP_ACTION_HISTORY_CONTRACT = (
    "full text of every non-empty steps/step_<id>.sh, ordered by numeric step ID"
)
RESULT_TASK_SOURCE = "result.json.task"
RESULT_ACTION_HISTORY_SOURCE = "result.json.action_history[].action before final '->'"
RESULT_ACTION_HISTORY_CONTRACT = (
    "every result.json action_history[].action in list order, with the final "
    "'->' and all following text removed"
)
RAW_RESULT_ACTION_HISTORY_SOURCE = "result.json.action_history"
RAW_RESULT_ACTION_HISTORY_CONTRACT = (
    "every non-empty result.json action_history entry in list order, unchanged"
)

_TRAILING_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class TaskArtifacts:
    """Everything the judge needs for one task, already read off disk."""

    task_id: str
    task_dir: str
    task: str
    action_history: list[str]
    screenshot_paths: list[str]

    @property
    def action_count(self) -> int:
        """Return the number of action-history entries."""
        return len(self.action_history)

    @property
    def screenshot_count(self) -> int:
        """Return the number of screenshots."""
        return len(self.screenshot_paths)


#: Reads one task directory's action history; see :func:`load_browser_step_actions`.
ActionHistoryLoader = Callable[[Path], list[str]]
#: Reads every task directory under a trajectories root.
ArtifactLoader = Callable[[Path], list[TaskArtifacts]]


class ArtifactLayout(str, Enum):
    """Selectable on-disk trajectory layouts, i.e. where a task's actions live.

    A layout only chooses the action history. The task text comes from
    ``task.json``/``result.json`` and the screenshots are always the root-level
    ``screenshots/*.png``, so the same run scored under two layouts differs only
    in what the judge reads as the agent's actions:

    - ``BROWSER_STEPS`` reads each ``browser-steps.jsonl`` row's
      natural-language ``action``, so the judge sees what the agent said it was
      doing. This is :data:`DEFAULT_ARTIFACT_SPEC`, what
      ``scripts/eval/persistent_cli.py`` scores.
    - ``STEP_SCRIPTS`` reads the full text of every ``steps/step_<id>.sh``, so
      the judge sees the shell commands actually executed. This is what a batch
      run scores by default.
    - ``RESULT_JSON`` reads ``result.json``'s ``action_history`` for runs that
      never wrote ``task.json`` (see ``result_action_history_mode`` for whether
      each entry is trimmed after its final ``->``).
    - ``AUTO`` keeps the legacy detection: ``RESULT_JSON`` when no task folder
      has a ``task.json``, otherwise ``STEP_SCRIPTS``. It never resolves to
      ``BROWSER_STEPS`` -- a directory holding both ``browser-steps.jsonl``
      and ``steps/`` is indistinguishable, and silently re-pointing existing
      ``auto`` callers at a different action source would move their scores.

    Which layout a benchmark run actually scores under is decided by the entry
    point, not by inspecting the run: ``scripts/eval/persistent_cli.py``
    (``runner.main``) is hardwired to :data:`DEFAULT_ARTIFACT_SPEC` and does not
    register ``--artifact-layout`` at all, while
    ``scripts/eval/persistent_cli_steps.py`` (``runner.layout_main``) exposes
    this enum and defaults to ``STEP_SCRIPTS``. ``run_online_mind2web_judge``
    passes no layout flag, so ``run.judge_script`` in the eval config picks the
    action source for a batch run -- and it defaults to the steps shim
    (``DEFAULT_JUDGE_SCRIPT`` in ``run/benchmarks/om2w.py``), i.e. to
    ``STEP_SCRIPTS``.
    """

    AUTO = "auto"
    BROWSER_STEPS = "browser-steps"
    STEP_SCRIPTS = "step-scripts"
    RESULT_JSON = "result-json"


def _screenshot_sort_key(path: Path) -> tuple[int, str]:
    """Order screenshots by their trailing step number, then by filename."""
    match = _TRAILING_NUMBER_RE.search(path.stem)
    return (int(match.group(1)) if match else 10**9, path.name.lower())


def load_screenshot_paths(path: Path) -> list[str]:
    """Return every root-level PNG in chronological browser-step order."""
    if not path.is_dir():
        return []
    screenshots = [item.resolve() for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png"]
    screenshots.sort(key=_screenshot_sort_key)
    return [str(item) for item in screenshots]


def _load_json_mapping(path: Path) -> dict[str, Any]:
    """Return the JSON object at ``path``, or an empty mapping if unusable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_description(payload: dict[str, Any]) -> str:
    """Return the task text from either descriptor key, or an empty string."""
    return str(payload.get("task") or payload.get("confirmed_task") or "").strip()


def load_browser_step_actions(task_dir: Path) -> list[str]:
    """Return the natural-language ``action`` of each ``browser-steps.jsonl`` row.

    This is the default :data:`ActionHistoryLoader`. Loaders receive the task
    directory rather than a specific file so that a layout is free to read any
    artifact under it.
    """
    path = task_dir / "browser-steps.jsonl"
    actions: list[str] = []
    if not path.exists():
        return actions

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            action = str(row.get("action") or "").strip()
            if action:
                actions.append(action)
    return actions


def load_step_action_history(task_dir: Path) -> list[str]:
    """Load each ordered ``steps/step_<id>.sh`` script as one action-history entry."""
    steps_dir = task_dir / "steps"
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


def load_task_artifacts(
    task_dir: Path,
    *,
    action_history_loader: ActionHistoryLoader = load_browser_step_actions,
) -> TaskArtifacts:
    """Read one ``task.json``-rooted task directory into :class:`TaskArtifacts`."""
    task_path = task_dir / "task.json"
    task_payload = _load_json_mapping(task_path)
    task_id = str(task_payload.get("task_id") or task_dir.name).strip()
    task = _task_description(task_payload)
    if not task:
        # The agent's workspace is rooted at the task directory, so a step that
        # writes its answer to "task.json" clobbers the harness descriptor. The
        # harness writes result.json after the episode with the same task text,
        # so recover from it instead of failing the whole judge run: one bad
        # directory used to abort discovery and leave every task unscored.
        result_payload = _load_json_mapping(task_dir / "result.json")
        task = _task_description(result_payload)
        task_id = str(result_payload.get("task_id") or task_id).strip()
    if not task:
        raise ValueError(f"{task_path}: missing task description")

    return TaskArtifacts(
        task_id=task_id,
        task_dir=str(task_dir.resolve()),
        task=task,
        action_history=action_history_loader(task_dir),
        screenshot_paths=load_screenshot_paths(task_dir / "screenshots"),
    )


def load_result_task_artifacts(
    task_dir: Path, *, trim_last_arrow: bool
) -> TaskArtifacts:
    """Read one ``result.json``-rooted task directory into :class:`TaskArtifacts`."""
    result_path = task_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    task_id = str(payload.get("task_id") or task_dir.name).strip()
    task = str(payload.get("task") or "").strip()
    if not task:
        raise ValueError(f"{result_path}: missing task description")

    return TaskArtifacts(
        task_id=task_id,
        task_dir=str(task_dir.resolve()),
        task=task,
        action_history=load_result_action_history(
            result_path, trim_last_arrow=trim_last_arrow
        ),
        screenshot_paths=load_screenshot_paths(task_dir / "screenshots"),
    )


def discover_artifacts(
    trajectories_dir: Path,
    *,
    marker: str,
    load_one: Callable[[Path], TaskArtifacts],
    on_duplicate: Literal["warn", "raise"] = "warn",
) -> list[TaskArtifacts]:
    """Load every task directory under ``trajectories_dir`` that contains ``marker``.

    Shared by every artifact layout; a layout supplies the marker filename that
    identifies a task directory and the per-directory reader. ``on_duplicate``
    selects what a repeated task ID does: ``warn`` keeps the first directory and
    skips the rest, ``raise`` reports every colliding ID.
    """
    # Symlinks are skipped because the agent's workspace is rooted at its own
    # task directory, so a step running e.g. `ln -sfn /workspace /workspace_backup`
    # drops a sibling link in the trajectories dir that points back at the task.
    # Path.is_dir() follows symlinks, so such a link used to be discovered as a
    # second copy of the same task.
    task_dirs = sorted(
        path
        for path in trajectories_dir.iterdir()
        if path.is_dir() and not path.is_symlink() and (path / marker).is_file()
    )

    artifacts: list[TaskArtifacts] = []
    seen: dict[str, TaskArtifacts] = {}
    duplicates: set[str] = set()
    for task_dir in task_dirs:
        artifact = load_one(task_dir)
        previous = seen.get(artifact.task_id)
        if previous is not None:
            duplicates.add(artifact.task_id)
            if on_duplicate == "warn":
                # Keep the first directory rather than aborting: a duplicate is
                # one unusable task, but raising leaves every task unscored.
                print(
                    f"[warn] duplicate task ID {artifact.task_id}: keeping "
                    f"{previous.task_dir}, skipping {artifact.task_dir}",
                    flush=True,
                )
            continue
        seen[artifact.task_id] = artifact
        artifacts.append(artifact)

    if duplicates and on_duplicate == "raise":
        raise ValueError(f"Duplicate task IDs: {sorted(duplicates)}")
    return artifacts


def discover_task_artifacts(
    trajectories_dir: Path,
    *,
    action_history_loader: ActionHistoryLoader = load_browser_step_actions,
) -> list[TaskArtifacts]:
    """Discover the ``task.json`` layout, the default for persistent-CLI runs."""
    return discover_artifacts(
        trajectories_dir,
        marker="task.json",
        load_one=partial(
            load_task_artifacts,
            action_history_loader=action_history_loader,
        ),
    )


def discover_result_task_artifacts(
    trajectories_dir: Path,
    *,
    trim_last_arrow: bool,
) -> list[TaskArtifacts]:
    """Discover the ``result.json`` layout, rejecting a run with no screenshots."""
    artifacts = discover_artifacts(
        trajectories_dir,
        marker="result.json",
        load_one=partial(load_result_task_artifacts, trim_last_arrow=trim_last_arrow),
        on_duplicate="raise",
    )
    # A missing screenshots/ directory is a layout mismatch, not a bad task:
    # judging would silently fall back to action text alone for the whole run.
    # Individual tasks with no captures stay tolerated.
    if artifacts and not any(item.screenshot_paths for item in artifacts):
        raise ValueError(
            f"{trajectories_dir}: no task directory contains root-level "
            "screenshots/*.png, so every task would be judged without images"
        )
    return artifacts


@dataclass(frozen=True)
class ArtifactSpec:
    """One on-disk trajectory layout: how to read it, and how to describe it.

    The ``*_source`` and ``*_contract`` strings are pure provenance — they are
    recorded in ``eval_manifest.json`` and on every result row so a scored run
    states which artifacts it actually read.
    """

    loader: ArtifactLoader = field(default=discover_task_artifacts)
    task_source: str = DEFAULT_TASK_SOURCE
    action_history_source: str = DEFAULT_ACTION_HISTORY_SOURCE
    action_history_contract: str = DEFAULT_ACTION_HISTORY_CONTRACT
    screenshot_source: str = DEFAULT_SCREENSHOT_SOURCE
    screenshot_contract: str = DEFAULT_SCREENSHOT_CONTRACT


DEFAULT_ARTIFACT_SPEC = ArtifactSpec()


def uses_result_json_layout(trajectories_dir: Path) -> bool:
    """Return whether the run looks like the result.json layout rather than task.json."""
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


def resolve_artifact_spec(
    trajectories_dir: Path,
    *,
    layout: ArtifactLayout | str = ArtifactLayout.AUTO,
    result_action_history_mode: str | None = None,
) -> ArtifactSpec:
    """Resolve an explicit artifact layout while retaining legacy auto-detection.

    ``result_action_history_mode`` only means anything to ``RESULT_JSON``; the
    other layouts ignore it, except that passing it forces ``AUTO`` to
    ``RESULT_JSON``.
    """
    layout = ArtifactLayout(layout)
    if layout is ArtifactLayout.AUTO:
        layout = (
            ArtifactLayout.RESULT_JSON
            if result_action_history_mode is not None
            or uses_result_json_layout(trajectories_dir)
            else ArtifactLayout.STEP_SCRIPTS
        )

    if layout is ArtifactLayout.BROWSER_STEPS:
        return DEFAULT_ARTIFACT_SPEC

    if layout is ArtifactLayout.RESULT_JSON:
        trim_last_arrow = result_action_history_mode != "raw"
        return ArtifactSpec(
            loader=partial(
                discover_result_task_artifacts,
                trim_last_arrow=trim_last_arrow,
            ),
            task_source=RESULT_TASK_SOURCE,
            action_history_source=(
                RESULT_ACTION_HISTORY_SOURCE
                if trim_last_arrow
                else RAW_RESULT_ACTION_HISTORY_SOURCE
            ),
            action_history_contract=(
                RESULT_ACTION_HISTORY_CONTRACT
                if trim_last_arrow
                else RAW_RESULT_ACTION_HISTORY_CONTRACT
            ),
        )

    return ArtifactSpec(
        loader=partial(
            discover_task_artifacts,
            action_history_loader=load_step_action_history,
        ),
        action_history_source=STEP_ACTION_HISTORY_SOURCE,
        action_history_contract=STEP_ACTION_HISTORY_CONTRACT,
    )
