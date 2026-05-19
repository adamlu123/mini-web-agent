#!/usr/bin/env python3
"""Compute cumulative token usage and cost per task up to a step budget N.

Per-step cumulative usage is parsed from `debug/steps.md`, which contains a
JSON blob per step with fields like:

    "model_usage": {
      "cumulative_request":  {"input_tokens": ...},
      "cumulative_response": {"input_tokens": ..., "output_tokens": ...,
                              "cached_input_tokens": ..., "total_tokens": ...}
    }

For a budget N we take the largest step <= N that has usage recorded.

Pricing (gpt-5.4, per 1M tokens, short context):
    non-cached input: $2.50
    cached input:     $0.25
    output:           $15.00
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

STEP_HEADER_RE = re.compile(r"^##\s*Step\s+(\d+)\s*$", re.MULTILINE)

DEFAULT_PRICE_INPUT_PER_M = 2.50
DEFAULT_PRICE_CACHED_INPUT_PER_M = 0.25
DEFAULT_PRICE_OUTPUT_PER_M = 15.00


def parse_steps_usage(md_path: Path) -> dict[int, dict]:
    """Return {step_num -> cumulative usage dict}."""
    if not md_path.exists():
        return {}
    text = md_path.read_text(encoding="utf-8", errors="replace")

    # Find all step headers and their positions.
    headers = [(int(m.group(1)), m.start()) for m in STEP_HEADER_RE.finditer(text)]
    if not headers:
        return {}
    headers.append((None, len(text)))

    out: dict[int, dict] = {}
    for i, (step, start) in enumerate(headers[:-1]):
        end = headers[i + 1][1]
        section = text[start:end]
        # Find the JSON block containing model_usage. Observation is in a
        # ```json ... ``` fence. Locate it by finding the first "{" after
        # "### Observation" and matching braces.
        obs_idx = section.find("### Observation")
        if obs_idx == -1:
            continue
        json_start = section.find("{", obs_idx)
        if json_start == -1:
            continue
        # Match braces.
        depth = 0
        in_str = False
        esc = False
        end_idx = None
        for j in range(json_start, len(section)):
            c = section[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end_idx = j + 1
                    break
        if end_idx is None:
            continue
        try:
            obj = json.loads(section[json_start:end_idx])
        except Exception:
            continue
        usage = obj.get("model_usage")
        if isinstance(usage, dict):
            out[step] = usage
    return out


def cumulative_at_budget(
    step_usage: dict[int, dict],
    budget: int,
) -> dict | None:
    """Return cumulative_response dict at the largest step <= budget."""
    eligible = sorted(s for s in step_usage.keys() if s <= budget)
    if not eligible:
        return None
    last = eligible[-1]
    u = step_usage[last].get("cumulative_response") or {}
    req = step_usage[last].get("cumulative_request") or {}
    return {
        "step": last,
        "input_tokens": int(u.get("input_tokens", 0)),
        "output_tokens": int(u.get("output_tokens", 0)),
        "total_tokens": int(u.get("total_tokens", 0)),
        "cached_input_tokens": int(u.get("cached_input_tokens", 0)),
        "reasoning_output_tokens": int(u.get("reasoning_output_tokens", 0)),
        "message_count": int(req.get("message_count", 0)),
    }


def compute_cost(usage: dict, *, p_in: float, p_cached: float, p_out: float) -> float:
    input_tok = usage["input_tokens"]
    cached = usage["cached_input_tokens"]
    non_cached_input = max(input_tok - cached, 0)
    output_tok = usage["output_tokens"]
    return (
        non_cached_input * p_in / 1_000_000
        + cached * p_cached / 1_000_000
        + output_tok * p_out / 1_000_000
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/luyadong/sandbox/mini-web-agent/outputs/default/0421_all_best_default_json_sum20_s300",
    )
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 150, 200, 250, 300])
    ap.add_argument("--output", default=None)
    ap.add_argument("--price-input", type=float, default=DEFAULT_PRICE_INPUT_PER_M, help="Input $/1M")
    ap.add_argument("--price-cached", type=float, default=DEFAULT_PRICE_CACHED_INPUT_PER_M, help="Cached-input $/1M")
    ap.add_argument("--price-output", type=float, default=DEFAULT_PRICE_OUTPUT_PER_M, help="Output $/1M")
    args = ap.parse_args()
    print(f"Pricing (per 1M): input=${args.price_input:.2f} cached=${args.price_cached:.2f} output=${args.price_output:.2f}")

    root = Path(args.root)
    task_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name != "config_snapshot"
    )

    per_task: list[dict] = []
    for td in task_dirs:
        usage_by_step = parse_steps_usage(td / "debug" / "steps.md")
        row: dict = {"task": td.name, "total_steps_with_usage": len(usage_by_step)}
        for b in args.budgets:
            u = cumulative_at_budget(usage_by_step, b)
            if u is None:
                row[f"step@{b}"] = None
                row[f"input@{b}"] = None
                row[f"output@{b}"] = None
                row[f"cached@{b}"] = None
                row[f"cost@{b}"] = None
            else:
                row[f"step@{b}"] = u["step"]
                row[f"input@{b}"] = u["input_tokens"]
                row[f"output@{b}"] = u["output_tokens"]
                row[f"cached@{b}"] = u["cached_input_tokens"]
                row[f"cost@{b}"] = compute_cost(u, p_in=args.price_input, p_cached=args.price_cached, p_out=args.price_output)
        per_task.append(row)

    print(f"Tasks: {len(per_task)}\n")
    print(f"{'budget':>7} | {'tasks':>5} | {'sum_input':>14} | {'sum_cached':>14} | "
          f"{'sum_output':>14} | {'avg_input':>10} | {'avg_output':>10} | "
          f"{'total_cost_usd':>14} | {'avg_cost_usd':>12}")
    for b in args.budgets:
        vals = [r for r in per_task if r[f"cost@{b}"] is not None]
        if not vals:
            continue
        sum_input = sum(r[f"input@{b}"] for r in vals)
        sum_output = sum(r[f"output@{b}"] for r in vals)
        sum_cached = sum(r[f"cached@{b}"] for r in vals)
        sum_cost = sum(r[f"cost@{b}"] for r in vals)
        n = len(vals)
        print(f"{b:>7} | {n:>5} | {sum_input:>14,} | {sum_cached:>14,} | "
              f"{sum_output:>14,} | {sum_input/n:>10,.0f} | {sum_output/n:>10,.0f} | "
              f"{sum_cost:>14,.2f} | {sum_cost/n:>12,.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = ["task", "total_steps_with_usage"]
        for b in args.budgets:
            header += [f"step@{b}", f"input@{b}", f"cached@{b}", f"output@{b}", f"cost@{b}"]
        with out.open("w") as f:
            f.write(",".join(header) + "\n")
            for r in per_task:
                row = [str(r.get(k, "")) if r.get(k) is not None else "" for k in header]
                f.write(",".join(row) + "\n")
        print(f"\nWrote per-task CSV: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
