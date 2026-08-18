"""Judge Online-Mind2Web trajectories with the upstream WebJudge evaluator.

This module is the judging engine: it fans tasks out across worker processes,
throttles requests to the model endpoint, retries a failed task, and writes a
manifest, a resumable results file, and a summary. Successful judge rows are
skipped on a subsequent invocation, while rows that contain ``evaluation_error``
are retried.

Reading trajectories off disk belongs to
:mod:`miniswewebagent.evaluation.om2w.artifacts`; this module only consumes the
:class:`~miniswewebagent.evaluation.om2w.artifacts.ArtifactSpec` it hands back.
The judging implementation and prompts come from the packaged adapter around the
upstream ``WebJudge_Online_Mind2Web_eval`` implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import random
import sys
import threading
import time
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from miniswewebagent.evaluation.om2w.artifacts import (
    DEFAULT_ARTIFACT_SPEC,
    ArtifactLayout,
    ArtifactSpec,
    TaskArtifacts,
    resolve_artifact_spec,
)
from miniswewebagent.evaluation.om2w.judge import (
    robust_webjudge_online_mind2web_eval,
)
from om2w_judge.utils import OpenaiEngine, extract_predication

MODE = "WebJudge_Online_Mind2Web_eval"
DEFAULT_TASKS_DIR = REPO_ROOT / "outputs/default/260717_persistent_cli_full/w150"


class ThrottledOpenaiEngine(OpenaiEngine):
    """Share a cross-process cap while retaining the requested worker count."""

    def __init__(
        self,
        *args: Any,
        request_semaphore: Any = None,
        default_max_output_tokens: int = 8192,
        **kwargs: Any,
    ) -> None:
        """Wrap the upstream engine with an optional shared request semaphore."""
        super().__init__(*args, **kwargs)
        self.request_semaphore = request_semaphore
        self.default_max_output_tokens = default_max_output_tokens

    def generate(self, *args: Any, **kwargs: Any) -> list[str]:
        """Generate one completion, blocking while the in-flight cap is reached."""
        if len(args) < 2 and "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = self.default_max_output_tokens
        if self.request_semaphore is None:
            return super().generate(*args, **kwargs)
        self.request_semaphore.acquire()
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.request_semaphore.release()



def output_results_path(output_path: Path, model: str, score_threshold: int) -> Path:
    """Return the resumable JSONL results file for this model/threshold pair."""
    return output_path / f"{MODE}_{model}_score_threshold_{score_threshold}_auto_eval_results.json"


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    """Return every well-formed JSON object in a results file, skipping bad lines."""
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
    """Return task IDs already scored without error, so a rerun can skip them."""
    completed: set[str] = set()
    for row in _read_result_rows(path):
        task_id = str(row.get("task_id") or "")
        if not task_id or row.get("evaluation_error"):
            continue
        if row.get("predicted_label") in (0, 1):
            completed.add(task_id)
    return completed


def append_result(path: Path, row: dict[str, Any], lock: Any) -> None:
    """Append one JSONL result row under ``lock``, shared across judge workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_subset(
    artifacts: Iterable[TaskArtifacts],
    args: argparse.Namespace,
    spec: ArtifactSpec,
    results_path: Path,
    completed: set[str],
    labels: Any,
    errors: Any,
    lock: Any,
    request_semaphore: Any,
) -> None:
    """Judge one worker's share of the tasks, appending a row per finished task.

    Each task is retried up to ``args.task_max_attempts`` times with backoff; a
    task that never succeeds gets an ``evaluation_error`` row so a later run
    retries just that one.
    """
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
                    "action_history_source": spec.action_history_source,
                    "screenshot_paths": artifact.screenshot_paths,
                    "screenshot_source": spec.screenshot_source,
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
            except Exception as exc:  # noqa: BLE001 - isolate and retry one judge task
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
                    "action_history_source": spec.action_history_source,
                    "screenshot_paths": artifact.screenshot_paths,
                    "screenshot_source": spec.screenshot_source,
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


def build_manifest(
    args: argparse.Namespace,
    artifacts: list[TaskArtifacts],
    workers: int,
    spec: ArtifactSpec,
) -> dict[str, Any]:
    """Describe the run and the artifacts it will judge, before any judging starts."""
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
            "task": spec.task_source,
            "action_history": spec.action_history_contract,
            "screenshots": spec.screenshot_contract,
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
    """Roll the results file up into counts, keeping only the last row per task."""
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


def parallel_eval(
    args: argparse.Namespace,
    *,
    spec: ArtifactSpec = DEFAULT_ARTIFACT_SPEC,
) -> None:
    """Judge every task under ``args.trajectories_dir``, resuming a partial run.

    Writes ``eval_manifest.json`` up front, appends judge rows to a resumable
    results file, and finishes with ``eval_summary.json``. ``spec`` selects the
    on-disk layout to read; it defaults to the ``task.json`` layout.
    """
    trajectories_dir = Path(args.trajectories_dir).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = spec.loader(trajectories_dir)
    if args.expected_tasks and len(artifacts) != args.expected_tasks:
        raise SystemExit(
            f"Expected {args.expected_tasks} task directories for the "
            f"{spec.task_source} layout, found {len(artifacts)}"
        )

    workers = max(1, min(args.num_worker, len(artifacts))) if artifacts else 0
    manifest = build_manifest(args, artifacts, workers, spec)
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
                spec,
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
                            spec,
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


def parse_args(
    description: str | None = None,
    *,
    include_artifact_layout: bool = False,
) -> argparse.Namespace:
    """Parse the judge CLI, resolving the API key from the environment if unset.

    ``include_artifact_layout`` adds the layout-selection flags; leave it off for
    an entry point that always reads one fixed layout.
    """
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
    # Both flags below only mean something to a caller that resolves a layout,
    # so they are registered together: advertising them on a single-layout CLI
    # would accept values that are then silently ignored.
    if include_artifact_layout:
        parser.add_argument(
            "--artifact-layout",
            choices=("auto", "browser-steps", "step-scripts", "result-json"),
            default="step-scripts",
            help=(
                "Artifact layout, i.e. which files are read as the action "
                "history. 'browser-steps' is the persistent-CLI default that "
                "scripts/eval/persistent_cli.py is hardwired to; 'auto' retains "
                "the legacy detection behavior and never picks it."
            ),
        )
        parser.add_argument(
            "--result_action_history_mode",
            "--result-action-history-mode",
            choices=("raw", "last-arrow"),
            default=None,
            help=(
                "For the result-json layout, load result.json action_history "
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


def main() -> None:
    """Judge a trajectories directory using the default ``task.json`` layout."""
    parallel_eval(parse_args())


def layout_main() -> None:
    """Judge a trajectories directory using a selected (or auto-detected) layout.

    The layout-aware entry point: adds ``--artifact-layout`` and
    ``--result_action_history_mode`` and resolves them into an ``ArtifactSpec``
    before judging.
    """
    args = parse_args(
        description=(
            "Run upstream WebJudge_Online_Mind2Web_eval over persistent-browser "
            "step scripts or result.json low-level actions."
        ),
        include_artifact_layout=True,
    )
    spec = resolve_artifact_spec(
        Path(args.trajectories_dir).resolve(),
        layout=ArtifactLayout(args.artifact_layout),
        result_action_history_mode=args.result_action_history_mode,
    )
    parallel_eval(args, spec=spec)


if __name__ == "__main__":
    main()
