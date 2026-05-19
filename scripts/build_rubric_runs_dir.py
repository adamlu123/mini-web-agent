#!/usr/bin/env python3
"""Adapter: convert mini-web-agent run outputs into the layout expected by
Odysseys/scripts/python/run_full_trajectory_per_rubric.py.

For each <task_id>/ under SRC, pick the latest final_runs/run_<NNN>/ folder
(by directory creation time / ctime), and emit:

    DST/<task_id>/
        steps.jsonl    # step 1 = full log (lines joined with "\n");
                       # extra steps carry remaining screenshots so that
                       # run_full_trajectory_per_rubric.py picks them all up.
        result.txt     # always "1.0\n"

Step 1 has the entire concatenated final_script_log.txt as its `action` and
the first screenshot. Each remaining screenshot becomes its own row with an
empty action so the upstream judge ingests every image as a screenshot asset
without polluting the textual action history (the rubric script skips rows
with no response/action when building action_history).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def latest_final_run(task_dir: Path) -> Path | None:
    runs_dir = task_dir / "final_runs"
    if not runs_dir.is_dir():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_ctime)


_SHOT_IDX = re.compile(r"final_execution_(\d+)_")


def ordered_screenshots(run_dir: Path) -> list[Path]:
    shots_dir = run_dir / "screenshots"
    if not shots_dir.is_dir():
        return []
    shots = list(shots_dir.iterdir())

    def key(p: Path) -> tuple[int, str]:
        m = _SHOT_IDX.search(p.name)
        return (int(m.group(1)) if m else 10**9, p.name)

    return sorted([p for p in shots if p.is_file()], key=key)


def build_one(task_dir: Path, dst_root: Path) -> tuple[bool, str]:
    run_dir = latest_final_run(task_dir)
    if run_dir is None:
        return False, "no final_runs/run_*"
    log_path = run_dir / "final_script_log.txt"
    if not log_path.is_file():
        return False, f"missing {log_path}"

    lines = [ln.rstrip("\n") for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    shots = ordered_screenshots(run_dir)
    action_text = "\n".join(lines)

    out_dir = dst_root / task_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_path = out_dir / "steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as f:
        if not shots:
            f.write(json.dumps({"step_num": 1, "action": action_text}, ensure_ascii=False) + "\n")
        else:
            first = {"step_num": 1, "action": action_text, "screenshot": str(shots[0].resolve())}
            f.write(json.dumps(first, ensure_ascii=False) + "\n")
            for i, shot in enumerate(shots[1:], start=2):
                f.write(json.dumps({"step_num": i, "action": "", "screenshot": str(shot.resolve())}, ensure_ascii=False) + "\n")

    (out_dir / "result.txt").write_text("1.0\n", encoding="utf-8")
    return True, f"{len(lines)} log lines, {len(shots)} screenshots, run={run_dir.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Run output dir containing <task_id>/final_runs/run_*/...")
    ap.add_argument("--dst", type=Path, required=True, help="Destination runs dir to create")
    ap.add_argument("--only-finished", action="store_true", help="Skip task dirs whose trajectory.json exit_status != Submitted")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    for task_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        if args.only_finished:
            traj = task_dir / "trajectory.json"
            try:
                es = (json.loads(traj.read_text()).get("info") or {}).get("exit_status")
            except Exception:
                es = None
            if es != "Submitted":
                n_skip += 1
                continue
        ok, msg = build_one(task_dir, args.dst)
        if ok:
            n_ok += 1
            print(f"[ok] {task_dir.name}: {msg}")
        else:
            n_skip += 1
            print(f"[skip] {task_dir.name}: {msg}")
    print(f"\nWrote {n_ok} task(s) to {args.dst}; skipped {n_skip}.")


if __name__ == "__main__":
    main()
