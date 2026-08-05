#!/usr/bin/env python3
"""Materialize per-budget snapshot folders of the selected final_runs/run_NNN.

For each budget N, for each task under --root, pick the largest run_NNN whose
first-mention step <= N (with mtime fallback), and symlink/copy it into:

    <out-parent>/<root-name>_N{N}/<task_id>/final_runs/run_NNN

By default also symlinks task.json and result.json alongside for convenience.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_by_step_budget import parse_run_steps, read_predicted_label  # noqa: E402


def pick_run_for_budget(task_dir: Path, run_steps: dict[int, int], budget: int) -> Path | None:
    final_runs = task_dir / "final_runs"
    if not final_runs.is_dir():
        return None
    existing: dict[int, Path] = {}
    for p in final_runs.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        tail = p.name.split("_", 1)[1]
        if tail.isdigit():
            existing[int(tail)] = p
    eligible = sorted(
        (rid for rid, step in run_steps.items() if step <= budget and rid in existing),
        reverse=True,
    )
    for rid in eligible:
        if read_predicted_label(existing[rid]) is not None:
            return existing[rid]
    # Fall back to largest eligible run even without judge_result.json.
    if eligible:
        return existing[eligible[0]]
    return None


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/luyadong/sandbox/mini-web-agent/outputs/default/0421_all_best_default_json_sum20_s300",
    )
    ap.add_argument(
        "--out-parent",
        default=None,
        help="Parent directory for the per-budget snapshots. Defaults to the parent of --root.",
    )
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 150, 200])
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink.")
    ap.add_argument(
        "--extra-files",
        nargs="*",
        default=["task.json", "result.json"],
        help="Additional per-task files to link alongside (if present).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_parent = Path(args.out_parent).resolve() if args.out_parent else root.parent
    task_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name != "config_snapshot"
    )

    for budget in args.budgets:
        out_root = out_parent / f"{root.name}_N{budget}"
        out_root.mkdir(parents=True, exist_ok=True)
        picked = 0
        missing = 0
        for td in task_dirs:
            run_steps = parse_run_steps(td)
            run_dir = pick_run_for_budget(td, run_steps, budget)
            if run_dir is None:
                missing += 1
                continue
            dst_run = out_root / td.name / "final_runs" / run_dir.name
            link_or_copy(run_dir.resolve(), dst_run, copy=args.copy)
            for fname in args.extra_files:
                src = td / fname
                if src.exists():
                    link_or_copy(src.resolve(), out_root / td.name / fname, copy=args.copy)
            picked += 1
        print(f"N={budget}: wrote {picked} tasks to {out_root} (missing={missing})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
