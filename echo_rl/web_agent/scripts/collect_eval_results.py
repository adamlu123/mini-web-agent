"""Consolidate web-agent eval-only runs into one markdown summary.

Scans an eval-outputs root for ``*/exports/dumped_evals/eval_only/
aggregated_results.jsonl`` files (written by ``echo_rl.web_agent.eval_entrypoint``)
and emits a single table of avg_score / pass@1 per run, plus the per-split
breakdown and a few rollout-health stats (avg turns/tokens, parse/env errors).

Usage:
    python -m echo_rl.web_agent.scripts.collect_eval_results \
        --root eval_outputs --out eval_outputs/RESULTS.md
"""

import argparse
import glob
import json
import os


def _pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "-"


def _count_rollouts(run_dir):
    return len(glob.glob(os.path.join(run_dir, "rollouts", "*", ".persistent_session.json")))


def _split_records(run_dir):
    out = {}
    for split in ("train", "val"):
        f = os.path.join(run_dir, "exports", "dumped_evals", "eval_only",
                         f"web_agent_om2w_easy_{split}.jsonl")
        if os.path.exists(f):
            out[split] = sum(1 for _ in open(f))
    return out


def collect(root):
    rows = []
    for agg in sorted(glob.glob(os.path.join(root, "*", "exports", "dumped_evals",
                                             "eval_only", "aggregated_results.jsonl"))):
        run = agg.split(os.sep)[len(root.rstrip(os.sep).split(os.sep))]
        run_dir = os.path.join(root, run)
        r = json.load(open(agg))
        rows.append({
            "run": run,
            "rollouts": _count_rollouts(run_dir),
            "splits": _split_records(run_dir),
            "train": r.get("eval/web_agent_om2w_easy_train/avg_score"),
            "val": r.get("eval/web_agent_om2w_easy_val/avg_score"),
            "all": r.get("eval/all/avg_score"),
            "avg_turns": r.get("eval/all/generate/avg_num_turns"),
            "avg_tokens": r.get("eval/all/generate/avg_num_tokens"),
            "parse_errors": r.get("eval/all/generate/parse_errors"),
            "env_errors": r.get("eval/all/generate/env_setup_errors"),
        })
    return rows


def to_markdown(rows):
    lines = [
        "| Run | Rollouts | train avg_score | val avg_score | **all avg_score** | avg turns | avg tokens | parse_err | env_err |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['rollouts']} | {_pct(r['train'])} | {_pct(r['val'])} | "
            f"**{_pct(r['all'])}** | "
            f"{r['avg_turns']:.1f} | {int(r['avg_tokens']) if r['avg_tokens'] else '-'} | "
            f"{r['parse_errors'] if r['parse_errors'] is not None else '-'} | "
            f"{r['env_errors'] if r['env_errors'] is not None else '-'} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_outputs")
    ap.add_argument("--out", default=None, help="write markdown table here (else stdout)")
    args = ap.parse_args()

    rows = collect(args.root)
    table = to_markdown(rows)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"[collect] wrote {len(rows)} runs -> {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
