# CUAWright contributor guide

Keep maintained code under `src/cuawright/`. Benchmark repositories, task data,
assets, VM images, websites, credentials, and results stay external.

## Core rules

- Keep the model/tool loop benchmark-neutral and easy to read. Surrounding
  providers, benchmark adapters, capabilities, scheduling, health, and
  provenance may remain substantial but must have explicit boundaries.
- Preserve OSWorld, WeaveBench, BenchCAD full/direct, and CADGenBench behavior.
- Treat each benchmark as a release-pinned external dependency behind its own
  adapter; do not silently patch its checkout or import another benchmark's
  private adapter implementation.
- Preserve exact model-visible prompts, messages, tools, ordering, budgets,
  compaction, submission, provider requests, and failure behavior during
  structural refactors. Record intentional behavior changes separately.
- Never expose credentials, evaluator answers, or hidden benchmark state.
- Preserve exact run provenance and immutable canonical baselines.
- Experiments use separate branches/worktrees and result roots. Keep the stable
  CLI small; do not turn every experimental variable into a production flag.
- Fail explicitly on setup, transport, persistence, and evaluation errors.

## Validation

Use the smallest tests covering each change, then run the complete maintained
suite, formatting, compile checks, golden request checks, and `git diff --check`.
Paid benchmark runs require explicit authorization and post-launch health checks.
