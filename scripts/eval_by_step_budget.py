#!/usr/bin/env python3
"""Evaluate success rate up to step budget N for each task.

For each task under a default-style output root:
  - Parse command_history.sh to find which trajectory step first mentions each
    final_runs/run_NNN. (These are the "# Step K" markers.)
  - For a given step budget B, pick the largest run_NNN whose first-mention
    step is <= B. Fall back to next lower run_NNN if that run has no
    judge_result.json.
  - Read predicted_label (1 = success, 0 = failure) from judge_result.json.

Report success rate aggregated across all tasks for each budget.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

RUN_RE = re.compile(r"run_(\d+)")
STEP_RE = re.compile(r"^#\s*Step\s+(\d+)\s*$", re.IGNORECASE)
STEP_FILE_RE = re.compile(r"^step_(\d+)\.sh$", re.IGNORECASE)


def parse_run_steps_from_history(history_path: Path) -> dict[int, int]:
    """Return {run_id -> first trajectory step that references it}."""
    result: dict[int, int] = {}
    if not history_path.exists():
        return result
    current_step = 0
    for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m_step = STEP_RE.match(line.strip())
        if m_step:
            current_step = int(m_step.group(1))
            continue
        for m in RUN_RE.finditer(line):
            rid = int(m.group(1))
            if rid not in result:
                result[rid] = current_step
    return result


def parse_run_steps_from_mtimes(task_dir: Path) -> dict[int, int]:
    """Fallback: use file mtimes to associate each run_NNN with a step."""
    steps_dir = task_dir / "steps"
    final_runs = task_dir / "final_runs"
    if not steps_dir.is_dir() or not final_runs.is_dir():
        return {}
    step_mtimes: list[tuple[int, float]] = []
    for name in sorted(steps_dir.iterdir()):
        m = STEP_FILE_RE.match(name.name)
        if not m:
            continue
        try:
            step_mtimes.append((int(m.group(1)), name.stat().st_mtime))
        except OSError:
            continue
    if not step_mtimes:
        return {}
    step_mtimes.sort(key=lambda x: x[0])
    out: dict[int, int] = {}
    for p in final_runs.iterdir():
        if not p.is_dir():
            continue
        m = RUN_RE.fullmatch(p.name)
        if not m:
            continue
        rid = int(m.group(1))
        jr = p / "judge_result.json"
        try:
            target = jr.stat().st_mtime if jr.exists() else p.stat().st_mtime
        except OSError:
            continue
        # Largest step whose mtime <= target.
        chosen = step_mtimes[0][0]
        for step_num, mt in step_mtimes:
            if mt <= target + 1.0:
                chosen = step_num
            else:
                break
        out[rid] = chosen
    return out


def parse_run_steps(task_dir: Path) -> dict[int, int]:
    history = parse_run_steps_from_history(task_dir / "command_history.sh")
    mtime_based = parse_run_steps_from_mtimes(task_dir)
    # Union: prefer history mapping when present; else mtime-based.
    merged = dict(mtime_based)
    merged.update(history)
    return merged


def read_predicted_label(run_dir: Path) -> int | None:
    jr = run_dir / "judge_result.json"
    if not jr.exists():
        return None
    try:
        data = json.loads(jr.read_text(encoding="utf-8"))
    except Exception:
        return None
    label = data.get("predicted_label")
    if label in (0, 1):
        return label
    return None


def pick_run_for_budget(
    task_dir: Path,
    run_steps: dict[int, int],
    budget: int,
) -> tuple[int | None, int | None]:
    """Return (run_id, predicted_label) for the chosen run at this budget."""
    final_runs = task_dir / "final_runs"
    if not final_runs.is_dir():
        return None, None
    # candidate runs: run_NNN with known step <= budget, that also physically exist
    existing = {
        int(p.name.split("_")[1]): p
        for p in final_runs.iterdir()
        if p.is_dir() and p.name.startswith("run_") and p.name.split("_")[1].isdigit()
    }
    eligible = sorted(
        (rid for rid, step in run_steps.items() if step <= budget and rid in existing),
        reverse=True,
    )
    for rid in eligible:
        label = read_predicted_label(existing[rid])
        if label is not None:
            return rid, label
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/luyadong/sandbox/mini-web-agent/outputs/default/0421_all_best_default_json_sum20_s300",
    )
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--output", default=None, help="Optional per-task CSV output path")
    args = ap.parse_args()

    root = Path(args.root)
    task_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name != "config_snapshot"
    )

    per_task: list[dict] = []
    for td in task_dirs:
        run_steps = parse_run_steps(td)
        row: dict = {"task": td.name, "run_steps": run_steps}
        for b in args.budgets:
            rid, label = pick_run_for_budget(td, run_steps, b)
            row[f"run@{b}"] = rid
            row[f"label@{b}"] = label
        per_task.append(row)

    print(f"Tasks: {len(per_task)}\n")
    for b in args.budgets:
        labels = [r[f"label@{b}"] for r in per_task]
        have = [x for x in labels if x is not None]
        successes = sum(1 for x in have if x == 1)
        no_run = sum(1 for x in labels if x is None)
        # Denominator = all tasks (tasks without any eligible run counted as failure)
        denom_all = len(per_task)
        print(
            f"Budget N<= {b:>3}: "
            f"success_rate(all)={successes}/{denom_all} = {successes/denom_all*100:.1f}% | "
            f"eligible_tasks={len(have)} (no_run={no_run}) | "
            f"success_rate(eligible)={successes}/{len(have) or 1} = "
            f"{(successes/len(have)*100) if have else 0:.1f}%"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            header = ["task"] + sum(
                ([f"run@{b}", f"label@{b}"] for b in args.budgets), []
            )
            f.write(",".join(header) + "\n")
            for r in per_task:
                vals = [r["task"]] + [
                    str(r[k]) if r.get(k) is not None else ""
                    for k in header[1:]
                ]
                f.write(",".join(vals) + "\n")
        print(f"\nWrote per-task CSV: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
