#!/usr/bin/env python3
"""End-to-end builder for a self-contained rubric-runs directory.

For each <task_id>/ under SRC, choose the run directory under
``final_runs/run_*`` as follows:

  1. The latest (by ctime) run that contains ``judge_result.json``.
  2. If none of the runs has ``judge_result.json``, fall back to the
     latest run overall.

Only ``screenshots/`` and ``final_script_log.txt`` are copied (no
symlinks), and a fresh ``steps.jsonl`` + ``result.txt`` are emitted
whose screenshot paths point at the newly copied files.

Output layout::

    DST/<task_id>/
        final_script_log.txt
        screenshots/<files>
        steps.jsonl       # absolute paths reference DST/<task_id>/screenshots/
        result.txt        # always "1.0\n"

Usage::

    python scripts/build_final_rubric_runs.py \
        --src  /path/to/run_outputs/<run_name> \
        --dst  /path/to/output_dir
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

_SHOT_IDX = re.compile(r"final_execution_(\d+)_")


def pick_run(task_dir: Path) -> Path | None:
    runs_dir = task_dir / "final_runs"
    if not runs_dir.is_dir():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None
    judged = [p for p in candidates if (p / "judge_result.json").is_file()]
    pool = judged if judged else candidates
    return max(pool, key=lambda p: p.stat().st_ctime)


def ordered_screenshots(shots_dir: Path) -> list[Path]:
    if not shots_dir.is_dir():
        return []

    def key(p: Path) -> tuple[int, str]:
        m = _SHOT_IDX.search(p.name)
        return (int(m.group(1)) if m else 10**9, p.name)

    return sorted([p for p in shots_dir.iterdir() if p.is_file()], key=key)


def build_one(task_dir: Path, dst_root: Path) -> tuple[bool, str]:
    run = pick_run(task_dir)
    if run is None:
        return False, "no final_runs/run_*"
    log_src = run / "final_script_log.txt"
    if not log_src.is_file():
        return False, f"missing {log_src}"

    out = dst_root / task_dir.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Copy final_script_log.txt
    shutil.copy2(log_src, out / "final_script_log.txt")

    # Copy screenshots/
    src_shots = run / "screenshots"
    dst_shots = out / "screenshots"
    if src_shots.is_dir():
        shutil.copytree(src_shots, dst_shots)
    shots = ordered_screenshots(dst_shots)

    # Build steps.jsonl referencing the *copied* screenshot paths.
    log_lines = [
        ln.rstrip("\n")
        for ln in (out / "final_script_log.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip()
    ]
    action_text = "\n".join(log_lines)

    steps_path = out / "steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as f:
        if not shots:
            f.write(json.dumps({"step_num": 1, "action": action_text}, ensure_ascii=False) + "\n")
        else:
            first = {
                "step_num": 1,
                "action": action_text,
                "screenshot": str(shots[0].resolve()),
            }
            f.write(json.dumps(first, ensure_ascii=False) + "\n")
            for i, shot in enumerate(shots[1:], start=2):
                f.write(
                    json.dumps(
                        {"step_num": i, "action": "", "screenshot": str(shot.resolve())},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    (out / "result.txt").write_text("1.0\n", encoding="utf-8")
    return True, f"{len(log_lines)} log lines, {len(shots)} screenshots, run={run.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Run output dir containing <task_id>/final_runs/run_*/...")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Destination directory to populate")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    for task_dir in sorted(p for p in args.src.iterdir() if p.is_dir() and len(p.name) == 40):
        ok, msg = build_one(task_dir, args.dst)
        if ok:
            n_ok += 1
            print(f"[ok]   {task_dir.name}: {msg}")
        else:
            n_skip += 1
            print(f"[skip] {task_dir.name}: {msg}")
    print(f"\nWrote {n_ok} task(s) to {args.dst}; skipped {n_skip}.")


if __name__ == "__main__":
    main()
