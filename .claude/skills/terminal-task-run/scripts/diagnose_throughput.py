#!/usr/bin/env python3
"""Explain where wall-clock time went in an RST batch.

Separates the three things that make a batch slow, which need different fixes:

  * episodes not converging  -> most tasks run to step_limit
  * slow model round trips   -> high seconds-per-step (gateway queueing at high
                                worker counts, or long observations)
  * environment overhead     -> wall time per task far above steps x per-step

    python diagnose_throughput.py --output-dir outputs/terminal/rst_g01 --workers 16
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, required=True, help="--workers the batch used.")
    ap.add_argument("--step-limit", type=int, default=80)
    args = ap.parse_args()

    root = args.output_dir.expanduser()
    rows = []
    for path in root.glob("*/summary.json"):
        r = json.loads(path.read_text())
        r["_dir"] = path.parent
        rows.append(r)
    if not rows:
        raise SystemExit(f"no summary.json under {root}")

    done = [r for r in rows if r.get("seconds") and not r.get("skipped")]
    secs = [r["seconds"] for r in done]
    steps = [r["n_steps"] for r in done if r.get("n_steps")]
    per_step = [r["seconds"] / r["n_steps"] for r in done if r.get("n_steps")]

    print(f"tasks finished        : {len(done)}")
    print(f"  skipped (compose)   : {sum(1 for r in rows if r.get('skipped'))}")
    print(f"  errored             : {sum(1 for r in rows if r.get('error'))}")

    print("\n--- convergence ---")
    exits = collections.Counter(r.get("exit_status") for r in done)
    for k, v in exits.most_common():
        print(f"  {str(k):16s} {v:5d}  ({100*v/len(done):.0f}%)")
    at_limit = sum(1 for s in steps if s >= args.step_limit)
    print(f"  at step_limit       : {at_limit}/{len(steps)} ({100*at_limit/max(len(steps),1):.0f}%)")
    if steps:
        print(f"  steps  median {st.median(steps):.0f}  p90 {pct(steps,.9):.0f}  max {max(steps)}")

    print("\n--- where the time went ---")
    builds = [r["build_seconds"] for r in done if r.get("build_seconds")]
    cached = sum(1 for r in done if r.get("image_was_cached"))
    if builds:
        share = sum(builds) / max(sum(secs), 1)
        print(f"  build   median {st.median(builds):.0f}s  total {sum(builds)/3600:.1f}h"
              f"  ({100*share:.0f}% of all task time)")
        print(f"  images served from cache: {cached}/{len(done)}")
    else:
        print("  build timings unavailable (run predates build_seconds instrumentation)")

    print("\n--- latency ---")
    if per_step:
        print(f"  sec/step  median {st.median(per_step):.1f}  p90 {pct(per_step,.9):.1f}")
    print(f"  sec/task  median {st.median(secs):.0f}  p90 {pct(secs,.9):.0f}  max {max(secs):.0f}")

    print("\n--- throughput ---")
    total = sum(secs)
    ideal = total / args.workers
    print(f"  summed task time    : {total/3600:.1f} worker-hours")
    print(f"  ideal wall at {args.workers:2d}w  : {ideal/3600:.1f} h")
    print(f"  -> if real wall clock greatly exceeds that, workers are starved:")
    print(f"     serialised image builds, host RAM pressure, or gateway concurrency limits.")

    print("\n--- scoring ---")
    scored = [r for r in done if isinstance(r.get("score"), int)]
    if scored:
        passed = sum(r["score"] for r in scored)
        print(f"  {passed}/{len(scored)} passed ({100*passed/len(scored):.0f}%)")
    null = sum(1 for r in done if r.get("score") is None)
    if null:
        print(f"  score=null (verifier produced no reward): {null}  <-- infra failure, investigate")

    # Where the steps go: reflection churn shows up as repeated self_reflect calls.
    reflect = []
    for r in done:
        steps_dir = r["_dir"] / "steps"
        if steps_dir.is_dir():
            n = sum(1 for p in steps_dir.glob("*.sh") if "python -m self_reflection" in p.read_text(errors="replace"))
            reflect.append(n)
    # Format-error retries are invisible in n_steps: a FormatError does not advance
    # the step counter, but it still costs a full model round trip. A prompt that
    # asks for one output shape while model.response_mode parses another shows up
    # here and nowhere else.
    interrupts = collections.Counter()
    for r in done:
        traj = r["_dir"] / "trajectory.json"
        if not traj.is_file():
            continue
        try:
            data = json.loads(traj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for msg in data.get("messages", []):
            kind = (msg.get("extra") or {}).get("interrupt_type")
            if kind:
                interrupts[kind] += 1
    if interrupts:
        print("\n--- retries and interrupts (not counted as steps) ---")
        for kind, n in interrupts.most_common():
            print(f"  {kind:26s} {n:6d}  ({n/len(done):.1f} per task)")
        if interrupts.get("FormatErrorRetry") or interrupts.get("FormatError"):
            print("  !! format errors mean the prompt's output shape and")
            print("     model.response_mode disagree. Every retry is a wasted round trip.")

    import shutil, subprocess as sp
    print("\n--- docker disk ---")
    try:
        out = sp.run(["docker", "system", "df"], capture_output=True, text=True, timeout=30).stdout
        print("  " + "\n  ".join(out.strip().splitlines()[:4]))
    except Exception as exc:  # noqa: BLE001
        print(f"  (docker system df unavailable: {exc})")
    usage = shutil.disk_usage("/")
    print(f"  root filesystem: {100*usage.used/usage.total:.0f}% used, "
          f"{usage.free/1e9:.0f} GB free")
    print("  (>85% used slows the daemon badly: build and exec both degrade)")

    if reflect:
        print("\n--- reflection churn ---")
        print(f"  self_reflect calls per task: median {st.median(reflect):.0f}  p90 {pct(reflect,.9):.0f}  max {max(reflect)}")
        print(f"  tasks calling it 3+ times  : {sum(1 for n in reflect if n >= 3)}/{len(reflect)}")
        print("  (repeated calls mean the gate keeps rejecting; each retry costs a full")
        print("   judge round trip plus the steps spent reacting to it)")


if __name__ == "__main__":
    main()
