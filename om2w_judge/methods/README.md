# `om2w_judge/methods/`

Online-Mind2Web judge implementations. This directory contains one
byte-for-byte official OSU source file plus local and legacy evaluator modes.
The functions do not all have the same sync/async API or return shape; consult
the selected module before calling it directly.

`om2w_judge/run.py` dispatches to these by `--mode`. Treat this directory as
mixed-source code: never edit the pinned official file, and prefer placing new
integration behavior in `src/miniswewebagent/evaluation/om2w/`.

## Status

| Module | Exported eval(s) | Status |
| --- | --- | --- |
| `webjudge_online_mind2web.py` | `WebJudge_Online_Mind2Web_eval` | **Official.** Verbatim OSU implementation used by the default evaluator through a hardened local adapter. |
| `webjudge_online_mind2web_sandbox.py` | `WebJudge_Online_Mind2Web_Sandbox_eval`, `..._WithThoughts_eval`, `..._ThoughtsOnly_eval` | **Local legacy extension.** Used by explicit sandbox modes and the iterative in-loop judge. |
| `webjudge_online_mind2web_sandbox_latest_run.py` | re-exports the three above | **Local legacy shim.** `run.py` imports the sandbox evals through it. |
| `agenttrek_eval.py`, `automomous_eval.py`, `webvoyager_eval.py`, `webjudge_general_eval.py` | one each | Other legacy judge modes dispatched by `run.py`; they are not used by the default Online-Mind2Web evaluator. |

## Official source boundary

`webjudge_online_mind2web.py` is copied without modification from
[OSU-NLP-Group/Online-Mind2Web at commit
`0e00e251`](https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/0e00e251cd32ac8f6aa9d08d9e3474b63bd02330/src/methods/webjudge_online_mind2web.py).
Its pinned properties are:

- size: 9,962 bytes;
- SHA-256: `e3cd499b7c1fc92cbd51d3c4216c95ebab774c2c2cd998c5ad6dad94710780c4`;
- no final newline, matching the official file.

Do not edit or format that file locally. `pyproject.toml` excludes it from Ruff,
and `tests/test_om2w_final_judge_messages.py` checks its size and checksum.
The official source assumes OSU's flat `src/` layout and imports `utils` as a
top-level module. `om2w_judge/__init__.py` supplies that import temporarily when
the vendored module loads, then restores any pre-existing global `utils` alias.

## The canonical path

`webjudge_online_mind2web.py` is the official judge this repo supports. Its
single public evaluator runs three stages:

1. `identify_key_points(task, model)` extracts the task's explicit requirements.
2. `judge_image(task, image_path, key_points, model)` scores each screenshot
   from 1–5.
3. `WebJudge_Online_Mind2Web_eval(...)` selects screenshots at or above
   `score_threshold`, caps them at `MAX_IMAGE`, and assembles the final verdict
   request from the task, numbered action history, normalized key points, and
   selected snapshot reasons.

The official module keeps its final system prompt and request construction as
function-local code. It does **not** export a prompt constant or a
`build_final_judge_messages(...)` helper.

Its inputs are **task + action history + screenshots**. It does not see the
agent's reasoning. Calling this function directly through `om2w_judge/run.py`
uses OSU's one-shot image scoring and parsing behavior.

`src/miniswewebagent/evaluation/om2w/judge.py` is the default local adapter. It
reuses the official `identify_key_points`, `judge_image`, `encode_image`, and
`MAX_IMAGE`, while adding:

- up to 10 attempts for each image;
- tolerant parsing of labeled text and JSON responses;
- retry detection for transient gateway responses;
- structured attempt and parse-failure records.

Because the official module exposes no final-message builder, the adapter
reconstructs that stage locally. Regression tests compare the complete official
and hardened requests, both with and without a selected screenshot, so prompt
or message-shape drift fails the test suite.

Both `scripts/eval/persistent_cli.py` and
`scripts/eval/persistent_cli_steps.py` call the same packaged runner and use
this adapter; they differ only in which artifacts they read as the action
history. `src/miniswewebagent/run/benchmarks/om2w.py` defaults to
`persistent_cli_steps.py`, i.e. to the `step-scripts` layout.

## Local legacy sandbox evals

`webjudge_online_mind2web_sandbox.py` imports the official `MAX_IMAGE`,
`identify_key_points`, and `judge_image` helpers, then wraps image scoring in
its own three-attempt parser/retry loop and replaces the final-verdict stage.
It adds:

- a `thoughts` parameter — the agent's reasoning as a judge input;
- three variants off one private core, selected by `include_action_history` /
  `include_thoughts` flags: actions only, actions + thoughts, thoughts only;
- three near-identical system prompts, one per variant, differing in which
  inputs they announce.

These prompts are not the official OSU prompt. The official evaluator has
criteria 1–7; its criterion 7 allows selecting a qualifying item without a
filter when the page already displays all available items. The sandbox prompts
replace that rule with stricter local criteria 7–8: use every available
filter/sort control, or directly verify a constraint from page or item details
when no control exists. Scores from the two policies are therefore not
necessarily comparable.

### Why it is not the default

- **No evaluation path reads it.** As of 2026-08-18, `run/benchmarks/om2w.py`
  only knows the `scripts/eval/` entry points into
  `src/miniswewebagent/evaluation/om2w/runner.py`, which use the hardened
  official-compatible adapter. The sandbox is reachable only by invoking
  `om2w_judge/run.py` by hand, or from the archived iterative agent.
- **No validated thoughts baseline is documented here.** Scores from
  `WithThoughts` or `ThoughtsOnly` should not be compared to the official
  evaluator's scores. Feeding the agent's own reasoning to the judge also
  changes the evidence being evaluated from outcome to intent plus outcome.
- **It has a different artifact contract.** `om2w_judge/run.py` starts from
  `result.json`, then resolves sandbox evidence from the latest usable
  `final_runs/run_N`, falling back to the task root. A `final_script_log.txt`
  overrides the result action history when it contains actions; screenshots are
  matching `screenshots/final_execution_*.png` files. The packaged evaluator
  instead uses the explicit layouts in
  `src/miniswewebagent/evaluation/om2w/artifacts.py`.
- **Three copies of one local prompt.** Editing a criterion means editing it
  three times. Its criteria 7–8 already differ materially from official OSU
  criterion 7, in addition to variant-specific input wording.

### What still references it

Legacy means "do not build the default evaluator on it", not deleted — these
still work:

- `om2w_judge/run.py` — modes `WebJudge_Online_Mind2Web_Sandbox_eval`,
  `..._eval_ctime` (same eval, screenshots sorted by creation time),
  `..._WithThoughts_eval`, `..._ThoughtsOnly_eval`.
- `src/miniswewebagent/agents/archive/iterative.py` — the archived iterative
  runner's in-loop judge.
- `src/miniswewebagent/tools/self_reflection.py` — ports the sandbox response
  parsers (a copy, not an import).
- `tests/test_webjudge_online_mind2web_sandbox_modes.py`.

### Migrating

| Instead of | Use |
| --- | --- |
| `--mode WebJudge_Online_Mind2Web_Sandbox_eval` via `om2w_judge/run.py` | `python scripts/eval/persistent_cli.py` |
| `judge_script: om2w_judge/run.py` in a config | drop the key — `run.judge_script` now only selects between the `scripts/eval/` entry points, and defaults to `persistent_cli_steps.py` |
| step-script or `result.json` trajectories | `python scripts/eval/persistent_cli_steps.py --artifact-layout ...` |
| `..._WithThoughts_eval` / `..._ThoughtsOnly_eval` | keep only for legacy reproduction; they have no documented validated baseline |

Scoring a run with the default packaged evaluator produces
`WebJudge_Online_Mind2Web_eval_<model>_score_threshold_<n>_auto_eval_results.json`
rather than the `..._Sandbox_eval_...` filename, so old and new results files
sit side by side without clobbering each other.
See `src/miniswewebagent/evaluation/README.md` for the evaluator's layouts,
flags, and output.
