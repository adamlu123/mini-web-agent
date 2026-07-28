"""Print total + per-level success for a persistent-CLI judge results file.

Usage:
  python scripts/spb_level_breakdown.py \
    --results-dir <RUN_ROOT>/outputs_eval_persistent \
    --tasks-file src/miniswewebagent/run/benchmarks/om2w_260220.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tasks-file", required=True)
    args = parser.parse_args()

    rows: dict[str, dict] = {}
    for path in glob.glob(os.path.join(args.results_dir, "*auto_eval_results.json")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task_id = row.get("task_id") or row.get("id")
                if task_id:
                    rows[task_id] = row

    tasks = json.load(open(args.tasks_file, encoding="utf-8"))
    items = tasks if isinstance(tasks, list) else tasks.get("tasks", [])
    level_of = {}
    for task in items:
        task_id = task.get("task_id") or task.get("id")
        level_of[task_id] = str(task.get("level") or task.get("difficulty") or "?").lower()

    total: dict[str, int] = defaultdict(int)
    success: dict[str, int] = defaultdict(int)
    for task_id, row in rows.items():
        level = level_of.get(task_id, "?")
        total[level] += 1
        if row.get("predicted_label") == 1:
            success[level] += 1

    judged = len(rows)
    passed = sum(1 for row in rows.values() if row.get("predicted_label") == 1)
    print(f"TOTAL: {passed}/{judged} = {passed / max(judged, 1) * 100:.1f}%")
    for level in ("easy", "medium", "hard", "?"):
        if total[level]:
            print(
                f"  {level}: {success[level]}/{total[level]}"
                f" = {success[level] / total[level] * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
