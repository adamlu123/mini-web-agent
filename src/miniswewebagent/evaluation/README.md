# `miniswewebagent.evaluation`

Scoring for completed agent runs. Nothing here drives a browser or calls an
agent model — it reads finished trajectories off disk and asks a judge model
whether each task succeeded.

Only Online-Mind2Web (`om2w/`) lives here today.

## `om2w/` — Online-Mind2Web WebJudge

Three modules, layered strictly downward (`runner` → `artifacts`, `runner` →
`judge`). Nothing imports back up.

| Module | Owns | Depends on |
| --- | --- | --- |
| `artifacts.py` | Reading trajectories off disk: the `TaskArtifacts` record, task discovery, the three on-disk layouts, and the `ArtifactSpec` that binds a layout to its provenance strings. | nothing in-package |
| `judge.py` | The judge call itself: image scoring with retries, response parsing, and a hardened wrapper around upstream `WebJudge_Online_Mind2Web_eval`. | `om2w_judge/` (vendored upstream) |
| `runner.py` | The engine: worker fan-out, request throttling, per-task retry, resumable results, manifest and summary, and the two CLI entry points. | `artifacts`, `judge` |

`runner` never opens a trajectory directory; it consumes whatever
`ArtifactSpec` it is handed. Adding a fourth layout means touching `artifacts.py`
only.

### On-disk layouts

A trajectories directory holds one subdirectory per task. Which files inside it
count as the action history depends on the layout:

| Layout | Marker file | Action history | Task text |
| --- | --- | --- | --- |
| `browser-steps` | `task.json` | every non-empty `.action` in `browser-steps.jsonl`, file order | `task.json.task` |
| `step-scripts` (batch default) | `task.json` | full text of each `steps/step_<id>.sh`, ordered by numeric step ID | `task.json.task` |
| `result-json` | `result.json` | each `action_history[].action`, truncated after its final `->` | `result.json.task` |

Every layout reads screenshots from `screenshots/*.png`, sorted by the trailing
number in the filename — **never** from `trajectory/`. `trajectory/` is a
post-run export that pads failed captures with blank placeholder PNGs to keep
its indices dense; feeding those to the judge shows it empty frames instead of
real page state.

Two discovery rules apply to all layouts, both learned from real runs:

- Symlinked subdirectories are skipped. The agent's workspace is rooted at its
  own task directory, so a step running `ln -sfn /workspace /workspace_backup`
  drops a sibling link pointing back at the task; `Path.is_dir()` follows it and
  the task gets discovered twice.
- A task directory whose `task.json` was clobbered by the agent falls back to
  `result.json` for the task text, rather than aborting the whole run.

### Running it

Two thin shims under `scripts/eval/` are the entry points; both take the same
core flags and differ only in whether they can select a layout.

```bash
# browser-steps layout (browser-steps.jsonl actions), no layout flags
python scripts/eval/persistent_cli.py \
  --trajectories_dir outputs/<run> --output_path outputs/<run>_eval

# layout-aware: --artifact-layout {auto,browser-steps,step-scripts,result-json}
python scripts/eval/persistent_cli_steps.py \
  --trajectories_dir outputs/<run> --output_path outputs/<run>_eval \
  --artifact-layout result-json
```

`--artifact-layout` defaults to `step-scripts` on this shim. `auto` picks
`result-json` when `--result_action_history_mode` was passed or when no task
directory has a `task.json`, and `step-scripts` otherwise — it never picks
`browser-steps`, since a run writes both `browser-steps.jsonl` and `steps/` and
changing what `auto` resolves to would move existing scores. Ask for
`--artifact-layout browser-steps` to score the default layout through this shim,
which is the same thing `persistent_cli.py` does with no flag at all. Pass
`--result_action_history_mode raw` to keep `action_history` entries untrimmed.

Useful flags: `--dry-run` writes the manifest and stops, which is the cheapest
way to confirm a layout was detected correctly; `--expected_tasks 0` disables
the hard-fail on task count, needed when scoring a `--limit`/`--task-level`
subset; `--num_worker` sets processes and `--max_in_flight` caps concurrent
judge requests across all of them.

Credentials resolve from `--api_key`, then `OPENAI_GATEWAY_API_KEY` /
`PHYAGI_API_KEY` / `OPENAI_API_KEY` when `--endpoint_target_uri` is set, else
`OPENAI_API_KEY`.

### Output

Written under `--output_path`:

- `eval_manifest.json` — the run's configuration plus an `artifact_contract`
  block naming exactly which files were read. Written before any judging, so a
  `--dry-run` produces it alone.
- `WebJudge_Online_Mind2Web_eval_<model>_score_threshold_<n>_auto_eval_results.json`
  — JSONL, one row per task, appended under a cross-process lock. Each row
  carries its own `action_history_source` / `screenshot_source`.
- `eval_summary.json` — counts and success rate over the last row per task.

Results are **resumable**: a rerun skips tasks that already have a row with a
`predicted_label` of 0 or 1, and retries rows carrying `evaluation_error`. To
force a full rescore, delete the results file.

### Callers

`miniswewebagent.utils.om2w_eval` builds the command line and post-processes the
results file; `run/benchmarks/om2w.py` invokes it after a generation batch when
`judge_enabled` is set; its `DEFAULT_JUDGE_SCRIPT` is `persistent_cli_steps.py`,
so a batch run with no `run.judge_script` scores the `step-scripts` layout. The
vendored upstream implementation lives in `om2w_judge/` at the repo root and is
reached via the `sys.path` insert at the top of `runner.py`.
