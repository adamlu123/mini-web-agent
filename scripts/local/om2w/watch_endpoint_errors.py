#!/usr/bin/env python
"""Report per-endpoint progress and gateway errors for a batch output directory.

Each task directory holds ``model_endpoint.json`` (the deployment pinned to that
task by the batch runner) and ``runtime_errors.jsonl`` (rate-limit / transient /
fatal gateway events logged by the model backend), so the two can be joined to
see whether a specific TRAPI deployment is being throttled.

    python scripts/local/om2w/watch_endpoint_errors.py outputs/default/<batch> [--watch 60]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

ERROR_EVENTS = ("rate_limit_error", "transient_gateway_error", "fatal_gateway_error")


def _iter_task_dirs(batch_dir: Path):
    for path in sorted(batch_dir.iterdir()):
        if path.is_dir() and (path / "model_endpoint.json").is_file():
            yield path


def collect(batch_dir: Path) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for task_dir in _iter_task_dirs(batch_dir):
        try:
            endpoint = json.loads((task_dir / "model_endpoint.json").read_text())["model_endpoint"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        row = stats[endpoint]
        row["tasks"] += 1
        if (task_dir / "result.json").is_file():
            row["finished"] += 1
        errors_path = task_dir / "runtime_errors.jsonl"
        if not errors_path.is_file():
            continue
        saw_rate_limit = False
        for line in errors_path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line).get("event", "")
            except json.JSONDecodeError:
                continue
            if event in ERROR_EVENTS:
                row[event] += 1
                saw_rate_limit = saw_rate_limit or event == "rate_limit_error"
        if saw_rate_limit:
            row["tasks_rate_limited"] += 1
    return stats


def render(batch_dir: Path) -> str:
    stats = collect(batch_dir)
    if not stats:
        return f"{batch_dir}: no task directories with model_endpoint.json yet"
    header = f"{'endpoint':<26} {'tasks':>6} {'done':>6} {'429s':>7} {'429 tasks':>10} {'transient':>10} {'fatal':>6}"
    lines = [f"{time.strftime('%H:%M:%S')}  {batch_dir}", header, "-" * len(header)]
    totals = defaultdict(int)
    for endpoint in sorted(stats):
        row = stats[endpoint]
        for key, value in row.items():
            totals[key] += value
        lines.append(
            f"{endpoint:<26} {row['tasks']:>6} {row['finished']:>6} "
            f"{row['rate_limit_error']:>7} {row['tasks_rate_limited']:>10} "
            f"{row['transient_gateway_error']:>10} {row['fatal_gateway_error']:>6}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<26} {totals['tasks']:>6} {totals['finished']:>6} "
        f"{totals['rate_limit_error']:>7} {totals['tasks_rate_limited']:>10} "
        f"{totals['transient_gateway_error']:>10} {totals['fatal_gateway_error']:>6}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds instead of printing once.")
    args = parser.parse_args()

    if args.watch <= 0:
        print(render(args.batch_dir))
        return
    while True:
        print(render(args.batch_dir), flush=True)
        print(flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
