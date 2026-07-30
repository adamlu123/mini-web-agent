"""Evaluate persistent-browser trajectories with Online-Mind2Web WebJudge.

This adapter is for runs produced by
``generation/best_default_judge_json_persistent_cli.yaml``.  For every task directory it
uses:

* ``task.json["task"]`` as the task description;
* every non-empty ``action`` in ``browser-steps.jsonl``, in file order, as the
  complete action history; and
* every PNG directly under ``screenshots/``, in browser-step order.

The judging implementation and prompts come from the existing
``eval_with_original_om2w.py`` wrapper around the upstream
``WebJudge_Online_Mind2Web_eval`` implementation.  The output is resumable:
successful judge rows are skipped on a subsequent invocation, while rows that
contain ``evaluation_error`` are retried.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import random
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the local engine first so eval_with_original_om2w's legacy absolute
# repository path cannot select the engine from a different worktree.
from om2w_judge.utils import OpenaiEngine, extract_predication  # noqa: E402

from eval_with_original_om2w import (  # noqa: E402
    robust_webjudge_online_mind2web_eval,
)


MODE = "WebJudge_Online_Mind2Web_eval"
DEFAULT_TASKS_DIR = REPO_ROOT / "outputs/default/260717_persistent_cli_full/w150"
DEFAULT_TASK_SOURCE = "task.json.task"
DEFAULT_ACTION_HISTORY_SOURCE = "browser-steps.jsonl.action"
DEFAULT_ACTION_HISTORY_CONTRACT = "every non-empty browser-steps.jsonl.action in file order"
DEFAULT_SCREENSHOT_SOURCE = "screenshots/*.png"
DEFAULT_SCREENSHOT_CONTRACT = "every root-level screenshots/*.png in browser-step order"
_TRAILING_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class TaskArtifacts:
    task_id: str
    task_dir: str
    task: str
    action_history: list[str]
    screenshot_paths: list[str]

    @property
    def action_count(self) -> int:
        return len(self.action_history)

    @property
    def screenshot_count(self) -> int:
        return len(self.screenshot_paths)


class ThrottledOpenaiEngine(OpenaiEngine):
    """Share a cross-process cap while retaining the requested worker count."""

    def __init__(
        self,
        *args: Any,
        request_semaphore: Any = None,
        default_max_output_tokens: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.request_semaphore = request_semaphore
        self.default_max_output_tokens = default_max_output_tokens

    def generate(self, *args: Any, **kwargs: Any) -> list[str]:
        if len(args) < 2 and "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = self.default_max_output_tokens
        if self.request_semaphore is None:
            return super().generate(*args, **kwargs)
        self.request_semaphore.acquire()
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.request_semaphore.release()


def _screenshot_sort_key(path: Path) -> tuple[int, str]:
    match = _TRAILING_NUMBER_RE.search(path.stem)
    return (int(match.group(1)) if match else 10**9, path.name.lower())


def load_action_history(path: Path) -> list[str]:
    """Return only the natural-language ``action`` field from each JSONL row."""
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


def load_screenshot_paths(path: Path) -> list[str]:
    """Return every root-level PNG in chronological browser-step order."""
    if not path.is_dir():
        return []
    screenshots = [item.resolve() for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png"]
    screenshots.sort(key=_screenshot_sort_key)
    return [str(item) for item in screenshots]


def load_task_artifacts(task_dir: Path) -> TaskArtifacts:
    task_path = task_dir / "task.json"
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    task_id = str(task_payload.get("task_id") or task_dir.name).strip()
    task = str(task_payload.get("task") or task_payload.get("confirmed_task") or "").strip()
    if not task:
        raise ValueError(f"{task_path}: missing task description")

    return TaskArtifacts(
        task_id=task_id,
        task_dir=str(task_dir.resolve()),
        task=task,
        action_history=load_action_history(task_dir / "browser-steps.jsonl"),
        screenshot_paths=load_screenshot_paths(task_dir / "screenshots"),
    )


def discover_task_artifacts(trajectories_dir: Path) -> list[TaskArtifacts]:
    task_dirs = sorted(
        path
        for path in trajectories_dir.iterdir()
        if path.is_dir() and (path / "task.json").is_file()
    )
    artifacts = [load_task_artifacts(path) for path in task_dirs]
    task_ids = [item.task_id for item in artifacts]
    if len(task_ids) != len(set(task_ids)):
        duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        raise ValueError(f"Duplicate task IDs: {duplicates}")
    return artifacts


def output_results_path(output_path: Path, model: str, score_threshold: int) -> Path:
    return output_path / f"{MODE}_{model}_score_threshold_{score_threshold}_auto_eval_results.json"


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def completed_task_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    for row in _read_result_rows(path):
        task_id = str(row.get("task_id") or "")
        if not task_id or row.get("evaluation_error"):
            continue
        if row.get("predicted_label") in (0, 1):
            completed.add(task_id)
    return completed


def append_result(path: Path, row: dict[str, Any], lock: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_subset(
    artifacts: Iterable[TaskArtifacts],
    args: argparse.Namespace,
    results_path: Path,
    completed: set[str],
    labels: Any,
    errors: Any,
    lock: Any,
    request_semaphore: Any,
) -> None:
    model = ThrottledOpenaiEngine(
        model=args.model,
        api_key=args.api_key,
        endpoint_target_uri=args.endpoint_target_uri,
        request_semaphore=request_semaphore,
        default_max_output_tokens=args.default_max_output_tokens,
    )

    for artifact in artifacts:
        if artifact.task_id in completed:
            print(f"[pid {os.getpid()}] skip completed {artifact.task_id}", flush=True)
            continue

        print(
            f"[pid {os.getpid()}] start {artifact.task_id}: "
            f"actions={artifact.action_count} screenshots={artifact.screenshot_count}",
            flush=True,
        )
        for task_attempt in range(1, args.task_max_attempts + 1):
            try:
                messages, input_text, system_msg, image_record, key_points = asyncio.run(
                    robust_webjudge_online_mind2web_eval(
                        artifact.task,
                        artifact.action_history,
                        artifact.screenshot_paths,
                        model,
                        args.score_threshold,
                    )
                )
                response = model.generate(messages, max_new_tokens=8192)[0]
                predicted_label = extract_predication(response, MODE)
                row = {
                    "task_id": artifact.task_id,
                    "mode": MODE,
                    "task_dir": artifact.task_dir,
                    "action_history": artifact.action_history,
                    "action_history_source": getattr(
                        args, "action_history_source", DEFAULT_ACTION_HISTORY_SOURCE
                    ),
                    "screenshot_paths": artifact.screenshot_paths,
                    "screenshot_source": getattr(
                        args, "screenshot_source", DEFAULT_SCREENSHOT_SOURCE
                    ),
                    "image_judge_record": image_record,
                    "key_points": key_points,
                    "input_text": input_text,
                    "system_msg": system_msg,
                    "evaluation_details": {
                        "response": response,
                        "predicted_label": predicted_label,
                    },
                    "predicted_label": predicted_label,
                    "task_attempts": task_attempt,
                }
                append_result(results_path, row, lock)
                labels.append(predicted_label)
                print(
                    f"[pid {os.getpid()}] done {artifact.task_id}: "
                    f"predicted_label={predicted_label} attempts={task_attempt}",
                    flush=True,
                )
                break
            except Exception as exc:
                if task_attempt < args.task_max_attempts:
                    delay = min(2 ** task_attempt + random.uniform(0, 5), 60)
                    print(
                        f"[pid {os.getpid()}] retry {artifact.task_id}: "
                        f"attempt={task_attempt}/{args.task_max_attempts} "
                        f"delay={delay:.1f}s error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue

                error_row = {
                    "task_id": artifact.task_id,
                    "mode": MODE,
                    "task_dir": artifact.task_dir,
                    "action_history": artifact.action_history,
                    "action_history_source": getattr(
                        args, "action_history_source", DEFAULT_ACTION_HISTORY_SOURCE
                    ),
                    "screenshot_paths": artifact.screenshot_paths,
                    "screenshot_source": getattr(
                        args, "screenshot_source", DEFAULT_SCREENSHOT_SOURCE
                    ),
                    "task_attempts": task_attempt,
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                    "evaluation_traceback": traceback.format_exc(),
                }
                append_result(results_path, error_row, lock)
                errors.append(artifact.task_id)
                print(
                    f"[pid {os.getpid()}] error {artifact.task_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )


def build_manifest(args: argparse.Namespace, artifacts: list[TaskArtifacts], workers: int) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": MODE,
        "model": args.model,
        "score_threshold": args.score_threshold,
        "num_workers": workers,
        "max_in_flight_requests": args.max_in_flight,
        "default_max_output_tokens": args.default_max_output_tokens,
        "task_max_attempts": args.task_max_attempts,
        "endpoint_mode": "gateway" if args.endpoint_target_uri else "openai",
        "trajectories_dir": str(Path(args.trajectories_dir).resolve()),
        "artifact_contract": {
            "task": getattr(args, "task_source", DEFAULT_TASK_SOURCE),
            "action_history": getattr(
                args, "action_history_contract", DEFAULT_ACTION_HISTORY_CONTRACT
            ),
            "screenshots": getattr(
                args, "screenshot_contract", DEFAULT_SCREENSHOT_CONTRACT
            ),
        },
        "n_tasks": len(artifacts),
        "n_actions": sum(item.action_count for item in artifacts),
        "n_screenshots": sum(item.screenshot_count for item in artifacts),
        "n_zero_action_tasks": sum(item.action_count == 0 for item in artifacts),
        "n_zero_screenshot_tasks": sum(item.screenshot_count == 0 for item in artifacts),
        "max_screenshots_per_task": max((item.screenshot_count for item in artifacts), default=0),
        "tasks": [
            {
                "task_id": item.task_id,
                "task_dir": item.task_dir,
                "action_count": item.action_count,
                "screenshot_count": item.screenshot_count,
            }
            for item in artifacts
        ],
    }


def summarize_results(results_path: Path, expected_task_ids: set[str]) -> dict[str, Any]:
    latest_by_task: dict[str, dict[str, Any]] = {}
    for row in _read_result_rows(results_path):
        task_id = str(row.get("task_id") or "")
        if task_id:
            latest_by_task[task_id] = row

    completed = [
        row
        for row in latest_by_task.values()
        if not row.get("evaluation_error") and row.get("predicted_label") in (0, 1)
    ]
    errors = [row for row in latest_by_task.values() if row.get("evaluation_error")]
    successes = sum(row.get("predicted_label") == 1 for row in completed)
    return {
        "mode": MODE,
        "result_file": str(results_path.resolve()),
        "expected_tasks": len(expected_task_ids),
        "rows_for_unique_tasks": len(latest_by_task),
        "completed_tasks": len(completed),
        "error_tasks": len(errors),
        "missing_tasks": sorted(expected_task_ids - set(latest_by_task)),
        "successful_tasks": successes,
        "success_rate_completed": successes / len(completed) if completed else 0.0,
    }


def parallel_eval(args: argparse.Namespace) -> None:
    trajectories_dir = Path(args.trajectories_dir).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = discover_task_artifacts(trajectories_dir)
    if args.expected_tasks and len(artifacts) != args.expected_tasks:
        raise SystemExit(
            f"Expected {args.expected_tasks} task directories with task.json, found {len(artifacts)}"
        )

    workers = max(1, min(args.num_worker, len(artifacts))) if artifacts else 0
    manifest = build_manifest(args, artifacts, workers)
    manifest_path = output_path / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}, indent=2), flush=True)

    if args.dry_run or not artifacts:
        print(f"Dry run complete. Manifest: {manifest_path}", flush=True)
        return

    results_path = output_results_path(output_path, args.model, args.score_threshold)
    completed = completed_task_ids(results_path)
    pending = [artifact for artifact in artifacts if artifact.task_id not in completed]
    print(
        f"Resuming with completed={len(completed)} pending={len(pending)} workers={workers}",
        flush=True,
    )

    if pending:
        workers = min(workers, len(pending))
        subsets = [pending[index::workers] for index in range(workers)]
        if workers == 1:
            labels: list[int] = []
            errors: list[str] = []
            request_semaphore = threading.BoundedSemaphore(args.max_in_flight)
            evaluate_subset(
                subsets[0],
                args,
                results_path,
                completed,
                labels,
                errors,
                threading.Lock(),
                request_semaphore,
            )
        else:
            with multiprocessing.Manager() as manager:
                labels = manager.list()
                errors = manager.list()
                lock = manager.Lock()
                request_semaphore = manager.BoundedSemaphore(args.max_in_flight)
                processes = [
                    multiprocessing.Process(
                        target=evaluate_subset,
                        args=(
                            subset,
                            args,
                            results_path,
                            completed,
                            labels,
                            errors,
                            lock,
                            request_semaphore,
                        ),
                    )
                    for subset in subsets
                ]
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()

    summary = summarize_results(results_path, {item.task_id for item in artifacts})
    summary_path = output_path / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args(description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Run upstream WebJudge_Online_Mind2Web_eval over persistent-browser "
            "browser-steps.jsonl actions and root screenshots."
        )
    )
    parser.add_argument("--trajectories_dir", default=str(DEFAULT_TASKS_DIR))
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="o4-mini")
    parser.add_argument("--score_threshold", type=int, default=3)
    parser.add_argument("--num_worker", type=int, default=150)
    parser.add_argument(
        "--max_in_flight",
        type=int,
        default=75,
        help="Cross-process cap on simultaneous judge requests; worker count is unchanged.",
    )
    parser.add_argument("--task_max_attempts", type=int, default=8)
    parser.add_argument("--default_max_output_tokens", type=int, default=8192)
    parser.add_argument("--expected_tasks", type=int, default=300)
    parser.add_argument(
        "--result_action_history_mode",
        "--result-action-history-mode",
        choices=("raw", "last-arrow"),
        default=None,
        help=(
            "For compatible evaluator variants, load result.json action_history "
            "unchanged or trim each entry after its final '->'."
        ),
    )
    parser.add_argument("--api_key", default="")
    parser.add_argument(
        "--endpoint_target_uri",
        "--endpoint-target-uri",
        dest="endpoint_target_uri",
        default=os.getenv("OPENAI_GATEWAY_ENDPOINT", ""),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.api_key:
        if args.endpoint_target_uri:
            args.api_key = (
                os.getenv("OPENAI_GATEWAY_API_KEY", "")
                or os.getenv("PHYAGI_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
            )
        else:
            args.api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.dry_run and not args.api_key:
        raise SystemExit("--api_key, OPENAI_GATEWAY_API_KEY, or OPENAI_API_KEY must be set")
    return args


if __name__ == "__main__":
    parallel_eval(parse_args())
