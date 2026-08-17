#!/usr/bin/env python
"""Poll an om2w batch output directory and report progress, cost and stalls.

Usage:
    python scripts/local/om2w/monitor_run.py OUTPUT_DIR [--tasks-csv CSV]
                                             [--interval SEC] [--log FILE]

Reports every INTERVAL seconds:
  * finished / in-flight / queued counts and exit-status breakdown
  * estimated spend (list-price assumption, see PRICES)
  * throughput and ETA
  * health warnings for tasks that stopped writing or that are stuck in an
    empty-response retry loop (the failure mode that silently burned $34 on
    m2w_exp_1557: the gateway returns empty text, the parse fails, and the
    agent retries without ever advancing api_calls).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

# $ per 1M tokens (input, cached input, output). Gateway billing may differ;
# token counts are exact, the dollar figures are an estimate.
PRICES = (1.25, 0.125, 10.00)

IDLE_WARN_SEC = 900
EMPTY_RATE_WARN = 0.5
EMPTY_MIN_RESPONSES = 20


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _task_cost(usage: dict) -> float:
    rate_in, rate_cached, rate_out = PRICES
    cached = usage.get("cached_input_tokens", 0)
    uncached = usage.get("input_tokens", 0) - cached
    return (uncached * rate_in + cached * rate_cached + usage.get("output_tokens", 0) * rate_out) / 1e6


def _newest_mtime(task_dir: Path) -> float:
    newest = 0.0
    for root, _, files in os.walk(task_dir):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                pass
    return newest


def _empty_response_rate(task_dir: Path) -> tuple[int, int]:
    path = task_dir / "raw_responses.jsonl"
    if not path.exists():
        return 0, 0
    total = empty = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not (record.get("raw_text") or "").strip():
                    empty += 1
    except OSError:
        return 0, 0
    return total, empty


def scan(output_dir: Path, expected_ids: set[str] | None) -> dict:
    now = time.time()
    finished: list[dict] = []
    running: list[dict] = []
    exits: dict[str, int] = {}

    for task_dir in sorted(output_dir.glob("m2w_exp_*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        if expected_ids is not None and task_id not in expected_ids:
            continue

        trajectory = _read_json(task_dir / "trajectory.json")
        usage = (trajectory.get("model", {}).get("usage", {}) or {}).get("cumulative_response", {})
        entry = {
            "task_id": task_id,
            "calls": int(trajectory.get("info", {}).get("api_calls", 0) or 0),
            "cost": _task_cost(usage) if usage else 0.0,
        }

        result_path = task_dir / "result.json"
        if result_path.exists():
            status = str(_read_json(result_path).get("exit_status", "") or "unknown")
            exits[status] = exits.get(status, 0) + 1
            finished.append(entry)
        else:
            entry["idle"] = now - _newest_mtime(task_dir)
            total, empty = _empty_response_rate(task_dir)
            entry["responses"] = total
            entry["empty_rate"] = (empty / total) if total else 0.0
            running.append(entry)

    return {"finished": finished, "running": running, "exits": exits}


def format_report(state: dict, *, total_tasks: int, started_at: float, baseline_done: int) -> str:
    finished = state["finished"]
    running = state["running"]
    done = len(finished)
    queued = max(0, total_tasks - done - len(running))
    spend = sum(t["cost"] for t in finished) + sum(t["cost"] for t in running)

    elapsed_h = (time.time() - started_at) / 3600
    completed_since = done - baseline_done
    rate = completed_since / elapsed_h if elapsed_h > 0.05 else 0.0
    eta = f"{(total_tasks - done) / rate:.1f}h" if rate > 0 else "n/a"

    lines = [
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {done}/{total_tasks} done | "
        f"{len(running)} running | {queued} queued | ${spend:,.2f} | "
        f"{rate:.0f} tasks/h | ETA {eta}",
        f"    exits: {state['exits'] or '{}'}",
    ]
    if finished:
        costs = [t["cost"] for t in finished]
        lines.append(
            f"    cost/task mean ${statistics.mean(costs):.2f} median ${statistics.median(costs):.2f}"
        )

    stalled = [t for t in running if t["idle"] > IDLE_WARN_SEC]
    looping = [
        t for t in running
        if t["responses"] >= EMPTY_MIN_RESPONSES and t["empty_rate"] >= EMPTY_RATE_WARN
    ]
    for task in sorted(stalled, key=lambda t: -t["idle"]):
        lines.append(
            f"    WARN stalled {task['task_id']} idle {task['idle'] / 60:.0f}min "
            f"at {task['calls']} calls (${task['cost']:.2f})"
        )
    for task in sorted(looping, key=lambda t: -t["cost"]):
        lines.append(
            f"    WARN empty-loop {task['task_id']} {task['empty_rate'] * 100:.0f}% empty "
            f"of {task['responses']} responses, {task['calls']} calls (${task['cost']:.2f})"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tasks-csv", type=Path, default=None,
                        help="Restrict monitoring to task ids in this CSV.")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    expected_ids = None
    total_tasks = 0
    if args.tasks_csv:
        with args.tasks_csv.open(newline="", encoding="utf-8-sig") as handle:
            expected_ids = {row["task_id"] for row in csv.DictReader(handle)}
        total_tasks = len(expected_ids)

    if not total_tasks:
        total_tasks = len(list(args.output_dir.glob("m2w_exp_*")))

    started_at = time.time()
    baseline_done = len(scan(args.output_dir, expected_ids)["finished"])

    handle = args.log.open("a", encoding="utf-8") if args.log else None
    try:
        while True:
            state = scan(args.output_dir, expected_ids)
            report = format_report(
                state,
                total_tasks=total_tasks,
                started_at=started_at,
                baseline_done=baseline_done,
            )
            print(report, flush=True)
            if handle:
                handle.write(report + "\n")
                handle.flush()

            if not state["running"] and len(state["finished"]) >= total_tasks:
                print("ALL TASKS COMPLETE", flush=True)
                if handle:
                    handle.write("ALL TASKS COMPLETE\n")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
