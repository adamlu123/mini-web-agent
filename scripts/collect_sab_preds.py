#!/usr/bin/env python3
"""Collect per-task final_script.py from a mini-web-agent SAB batch run into a
ScienceAgentBench `pred_programs/` directory for scoring by the SAB harness.

The om2w batch runner writes each task's workspace at
`<batch_output_dir>/<task_id>/`, so the agent's final artifact is
`<batch_output_dir>/sab_<iid>/final_script.py`. ScienceAgentBench's evaluator
expects `pred_programs/pred_<gold_program_name>` per instance. This script maps
one to the other, writing a literal "ERROR" for any task with no final script
(matching SAB agent.py behaviour for unparseable outputs, so the instance counts
as an invalid program rather than crashing the harness).

It also emits a minimal run log jsonl (one `{"history": [...], "cost": 0.0}` line
per task, in SAB verified order) so `calculate_metrics.py` can pair runs.

Usage:
  python scripts/collect_sab_preds.py \
    --batch-dir outputs/sab_science_sft/<batch_name> \
    --tasks-file src/miniswewebagent/run/benchmarks/sab_verified.json \
    --pred-out /data/t-yifeili/ScienceAgentBench/pred_programs \
    --run-log  /data/t-yifeili/ScienceAgentBench/sab_sft_run.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument(
        "--tasks-file",
        default=str(
            Path(__file__).resolve().parent.parent
            / "src/miniswewebagent/run/benchmarks/sab_verified.json"
        ),
    )
    ap.add_argument("--pred-out", required=True)
    ap.add_argument("--run-log", default="")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    tasks = json.loads(Path(args.tasks_file).read_text())
    pred_out = Path(args.pred_out)
    pred_out.mkdir(parents=True, exist_ok=True)

    run_log_lines = []
    n_ok = n_missing = 0
    for t in tasks:
        gold = t["sab"]["gold_program_name"]
        script = batch_dir / t["task_id"] / "final_script.py"
        dest = pred_out / ("pred_" + gold)
        if script.is_file() and script.read_text(encoding="utf-8").strip():
            dest.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
            n_ok += 1
        else:
            dest.write_text("ERROR", encoding="utf-8")
            n_missing += 1
            print(f"[missing] {t['task_id']} -> {dest.name} (wrote ERROR placeholder)")
        run_log_lines.append(json.dumps({"history": [], "cost": 0.0}))

    if args.run_log:
        Path(args.run_log).write_text("\n".join(run_log_lines) + "\n", encoding="utf-8")

    print(
        f"收集完成: {n_ok}/{len(tasks)} 有 final_script.py, {n_missing} 缺失(ERROR占位) -> {pred_out}"
        + (f"; run log -> {args.run_log}" if args.run_log else "")
    )


if __name__ == "__main__":
    main()
