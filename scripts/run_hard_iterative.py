"""Parallel batch runner for iterative agent across all hard tasks.

Usage:
    cd /home/luyadong/sandbox/mini-web-agent
    source /home/luyadong/cred.sh
    python scripts/run_hard_iterative.py \
        --output-dir outputs/iterative_hard \
        --num-workers 20 \
        [--task-ids id1 id2 ...]   # optional: run subset

Outputs:
    outputs/iterative_hard/<task_id>_<timestamp>/   per-task run dir
    outputs/iterative_hard/batch_results.jsonl      one JSON line per task
    outputs/iterative_hard/batch_summary.json       aggregate success rate
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from miniswewebagent.run.iterative import run_one  # noqa: E402
from miniswewebagent.tasks.om2w import load_om2w_task  # noqa: E402

TASKS_FILE = REPO_ROOT / "src" / "miniswewebagent" / "run" / "benchmarks" / "om2w_260220.json"
BASE_CONFIGS = [
    "mini.yaml",
    "iterative.yaml",
]


def _load_hard_task_ids(tasks_file: Path) -> list[str]:
    tasks = json.loads(tasks_file.read_text())
    return [t["task_id"] for t in tasks if t.get("level") == "hard"]


def _run_one_task(
    task_id: str,
    output_dir: Path,
    results_path: Path,
    lock: threading.Lock | multiprocessing.Lock,
    already_done: set[str],
) -> dict:
    if task_id in already_done:
        print(f"[SKIP] {task_id} already done")
        return {}

    print(f"[START] {task_id}")
    result: dict = {"task_id": task_id, "started_at": datetime.now().isoformat()}
    try:
        r = run_one(
            task_id=task_id,
            tasks_file=TASKS_FILE,
            config_spec=BASE_CONFIGS,
            output_dir=output_dir,
            snapshot_config=True,
        )
        result["success"] = bool(r.get("_iterative_success", False))
        result["rounds"] = r.get("_iterative_rounds", 0)
        result["final_round_dir"] = r.get("_final_round_dir", "")
        result["exit_status"] = r.get("exit_status", "")
        result["output_dir"] = r.get("_output_dir", "")
    except Exception as exc:
        result["success"] = False
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(f"[ERROR] {task_id}: {exc}")

    result["finished_at"] = datetime.now().isoformat()
    success_marker = "✓" if result.get("success") else "✗"
    print(f"[{success_marker}] {task_id}  rounds={result.get('rounds','?')}")

    with lock:
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    return result


def _worker_main(task_ids: list[str], output_dir: Path, results_path: Path, lock, already_done: set[str]) -> None:
    for task_id in task_ids:
        _run_one_task(task_id, output_dir, results_path, lock, already_done)


def _load_already_done(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("task_id"):
                done.add(obj["task_id"])
        except Exception:
            pass
    return done


def _write_summary(results_path: Path, output_dir: Path) -> None:
    if not results_path.exists():
        return
    results: list[dict] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    errors = sum(1 for r in results if "error" in r)
    summary = {
        "total": total,
        "successes": successes,
        "failures": total - successes - errors,
        "errors": errors,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "generated_at": datetime.now().isoformat(),
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== BATCH SUMMARY ===")
    print(f"Total: {total}  |  Success: {successes}  |  Rate: {summary['success_rate']*100:.1f}%")


def main() -> None:
    global TASKS_FILE
    parser = argparse.ArgumentParser(description="Run all hard iterative tasks in parallel.")
    parser.add_argument("--output-dir", default="outputs/iterative_hard", help="Base output directory.")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of parallel worker processes.")
    parser.add_argument("--task-ids", nargs="*", help="Subset of task IDs to run (default: all hard).")
    parser.add_argument("--tasks-file", default=str(TASKS_FILE), help="Path to tasks JSON file.")
    args = parser.parse_args()

    TASKS_FILE = Path(args.tasks_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "batch_results.jsonl"

    task_ids = args.task_ids or _load_hard_task_ids(TASKS_FILE)
    already_done = _load_already_done(results_path)

    pending = [t for t in task_ids if t not in already_done]
    print(f"Tasks total={len(task_ids)}  already_done={len(already_done)}  pending={len(pending)}")

    if not pending:
        print("Nothing to do.")
        _write_summary(results_path, output_dir)
        return

    num_workers = max(1, min(args.num_workers, len(pending)))
    print(f"Launching {num_workers} worker processes...")

    # Split tasks across workers
    subsets = [pending[i::num_workers] for i in range(num_workers)]
    subsets = [s for s in subsets if s]

    lock = multiprocessing.Lock()
    with multiprocessing.Manager() as manager:
        processes: list[multiprocessing.Process] = []
        for subset in subsets:
            p = multiprocessing.Process(
                target=_worker_main,
                args=(subset, output_dir, results_path, lock, already_done),
                daemon=False,
            )
            p.start()
            processes.append(p)
            print(f"  Worker PID={p.pid} handling {len(subset)} tasks")

        for p in processes:
            p.join()

    print("All workers finished.")
    _write_summary(results_path, output_dir)


if __name__ == "__main__":
    main()
