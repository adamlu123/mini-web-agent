# Pipeline Report: `0421_all_best_default_json_sum20_s300_final_resp`

This report describes end-to-end what was executed when the run with output
directory
`outputs/default/0421_all_best_default_json_sum20_s300_final_resp` was launched
from config
[src/miniswewebagent/config/best_default_judge_json.yaml](../src/miniswewebagent/config/best_default_judge_json.yaml).
It covers the launch command, the config's effect on agent behavior, the
per-task runtime loop, the artifacts produced per task, and the separate
offline WebJudge evaluation that scores the resulting directory.

Companion docs (still current):
[execution-flow.md](execution-flow.md),
[step-llm-execution-flow.md](step-llm-execution-flow.md),
[step_budget_accuracy.md](step_budget_accuracy.md).

---

## 1. Launch command

Taken verbatim from [scripts/run_command.sh](../scripts/run_command.sh#L56-L60):

```bash
source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0421_all_best_default_json_sum20_s300_final_resp
```

Key parameters:

- Entry point: [src/miniswewebagent/run/benchmarks/om2w.py](../src/miniswewebagent/run/benchmarks/om2w.py)
  (benchmark runner wrapping the single-task runner in
  [src/miniswewebagent/run/mini.py](../src/miniswewebagent/run/mini.py)).
- Config: `best_default_judge_json.yaml` merged on top of the default
  `mini.yaml`.
- Tasks file: `src/miniswewebagent/run/benchmarks/om2w_260220.json` (from
  `run.tasks_file` in the YAML) — the full Online-Mind2Web 260-task set
  (`--task-level all`).
- Concurrency: 300 worker processes (one thread-pool/process per task, capped
  at 300 concurrently).
- `cred.sh` injects the Browserbase + PhyAGI gateway credentials (`BROWSERBASE_API_KEY`,
  `BROWSERBASE_PROJECT_ID`, gateway auth) consumed by the Playwright cloud
  session and the model gateway.

The benchmark runner creates the output directory, snapshots the resolved
config specs into `config_snapshot/`, and spawns one child invocation per task.
Each child runs `miniswewebagent.run.mini.run_one` with the task id and a
per-task subdirectory (`<output-dir>/<task_id>/`).

The snapshot is visible in the output: `config_snapshot/config_spec_manifest.json`
records that the run was driven by the single file
`best_default_judge_json.yaml`, saved as
`config_snapshot/00_best_default_judge_json.yaml`.

---

## 2. What the config does (distinctive choices)

`best_default_judge_json.yaml` is the JSON-response variant of
`best_default_judge.yaml`. Relative to the baseline it flips a handful of
important switches:

### 2.1 Model contract — strict JSON step

- `model.response_mode: json_schema` → the model MUST return a single JSON
  object per turn with exactly four fields:
  `{"thought", "bash_command", "done", "final_response"}`. No XML, no prose,
  no fences. The bash command is a single shell string per step; Python is
  invoked via `python - <<'PY' ... PY` heredocs inside that one command.
- `model.max_output_tokens: 16000`.
- `model.attach_observation_screenshot: false` → unlike the browser-centric
  default agent, **step screenshots are NOT auto-attached to the next
  prompt**. The agent has to call `miniswewebagent.tools.image_qa` explicitly
  whenever it needs visual grounding.
- `model.format_error_template` explains exactly how to recover from a JSON
  parse error.
- Underlying gateway (inherited from `mini.yaml` and `environment.env`):
  `OPENAI_GATEWAY_ENDPOINT=http://gateway.phyagi.net/api/responses`,
  `OPENAI_GATEWAY_MODEL=gpt-5.4`.

### 2.2 Environment — local terminal workspace, not a live browser

- `environment.environment_class: local_workspace` → this is **not** the
  `local_browser` environment from `execution-flow.md`. There is no
  Playwright runtime driven by the harness. Instead, each step is just a
  bash command executed in the per-task workspace directory, with a 240 s
  `command_timeout_seconds` and `output_truncation_chars: 24000`.
- `environment.credentials_file: /home/luyadong/cred.sh` is sourced before
  every command so the agent can spawn its own Browserbase sessions
  from inside heredocs (the system prompt template includes a full example).
- `environment.env` forces non-interactive output (`PAGER=cat`, `TQDM_DISABLE=1`,
  `PIP_PROGRESS_BAR=off`), which keeps command output deterministic.

### 2.3 Agent — default, single-pass, summary-compacted, judge-gated

- `agent.agent_class: default` → the classic "one step per model turn" loop
  from [src/miniswewebagent/agents/default.py](../src/miniswewebagent/agents/default.py).
  No iterative explorer/actor split.
- `agent.step_limit: 300` → up to 300 bash steps per task (matching the
  `s300` in the output directory name).
- `agent.summary_every_n_steps: 20` → every 20 model calls the agent
  rewrites its conversation history into a single summary message
  (`_compact_history`), keeping the context window bounded across long
  tasks. This is the `sum20` in the directory name.
- `agent.require_self_reflection_success: true` → the agent cannot emit
  `done: true` unless its most recent `final_runs/run_<id>/judge_result.json`
  has `predicted_label: 1`. The gate check lives in `_self_reflection_gate_error`
  and is enforced inside `execute_actions` before a `done` turn is accepted
  as an exit.

### 2.4 Built-in self-reflection ("judge inside the run")

The system prompt is long, but it pins the agent to a very specific per-task
workflow (not a free-form loop):

1. Parse the task into `plan.md` as a numbered checklist of critical points.
2. Author `judge_config.json` once with four prompts for the
   `miniswewebagent.tools.self_reflection` CLI
   ([src/miniswewebagent/tools/self_reflection.py](../src/miniswewebagent/tools/self_reflection.py)).
3. Explore the live site from Browserbase sessions, using
   `miniswewebagent.tools.image_qa` for screenshot QA.
4. Write `final_script.py` and execute it inside a fresh
   `final_runs/run_<id>/` subfolder. That script must:
   - create/reset `final_runs/run_<id>/final_script_log.txt`,
   - write `step <N> action: ...` for every constraint-relevant action,
   - save per-critical-point screenshots to
     `final_runs/run_<id>/screenshots/final_execution_<N>_<action>.png`.
5. Run
   `python -m miniswewebagent.tools.self_reflection --config judge_config.json --workspace-dir <ws> --output final_runs/run_<id>/judge_result.json`.
   The tool scores every screenshot once per critical-point list (Stage 1),
   then aggregates all per-image reasonings plus the run log into a single
   `Status: success|failure` verdict (Stage 2).
6. Only after `judge_result.json.predicted_label == 1` may the agent emit
   `done: true` with a non-empty `final_response`.

The `_resp` suffix in the output directory reflects the explicit emphasis in
this config on producing a clean `final_response` datum (price/code/count/etc.)
alongside the verified screenshots.

### 2.5 Run-level post-hoc evaluation settings

The `run:` block still declares an external post-hoc judge, which the
benchmark runner invokes after each task completes:

- `run.judge_enabled: true`
- `run.judge_model: o4-mini`
- `run.judge_script: /home/luyadong/sandbox/mini-web-agent/om2w_judge/run.py`

This is separate from the in-agent `self_reflection` tool (see §4).

---

## 3. What happens per task

For each task in `om2w_260220.json`, the runner:

1. Creates `outputs/default/0421_all_best_default_json_sum20_s300_final_resp/<task_id>/`
   and drops `task.json` (the task metadata).
2. Boots the `default` agent with the system/instance templates rendered
   against that task's `task`, `task_id`, `start_url`, and workspace paths.
3. Runs the step loop up to 300 model calls. Each step:
   - Model returns the JSON object.
   - Harness parses `bash_command`, writes the command to
     `steps/step_XXXX.py` (and mirrors into `raw_responses.jsonl` /
     `trajectory.json`), executes it with 240 s timeout, captures stdout/stderr.
   - Observation is rendered from `environment.observation_template` and fed
     back as the next user message.
   - Every 20 calls the history is compacted by `_compact_history`.
4. When the model returns `done: true`, the gate in `execute_actions`
   re-reads the latest `final_runs/run_<id>/judge_result.json`; the step is
   only accepted as an exit if `predicted_label == 1`. Otherwise the model
   gets a `SelfReflectionGate` error message and must continue iterating.
5. On exit (success, step limit, or exception) the runner writes
   `result.json` with `task_id`, `task`, `start_url`, `action_history`
   (reconstructed from debug steps), `thoughts`, `final_result_response`,
   `submission`, `exit_status`, and copies of the trajectory screenshots
   into `trajectory/`.

Typical per-task artifact layout (as seen in
`0059adc6b12a3822305deb68929b2de8/`):

```text
<task_id>/
  task.json
  plan.md
  judge_config.json
  steps/step_0001.py ...
  raw_responses.jsonl
  trajectory.json
  trajectory/0_full_screenshot.png ...
  screenshots/
  debug/steps/step_0001.json ...
  explore_*.txt                 # agent's own exploration scratch notes
  final_script.py               # last canonical script (copied from latest run)
  final_runs/
    run_001/ ... run_017/
      final_script.py
      final_script_log.txt
      screenshots/final_execution_1_*.png ...
      judge_result.json         # written by self_reflection
  logs/
  result.json                   # final harness export
```

`final_runs/run_001 ... run_NNN` captures the natural iteration inside a
single task: the agent tries the script, `self_reflection` fails or
parse-fails, the agent patches the script, reruns in a new `run_<id+1>/`,
and re-invokes `self_reflection`. The example task above went through 17
iterations before the judge passed.

`result.json.action_history` is reconstructed from `debug/steps/` by the
harness and is what downstream evaluators consume. `final_result_response`
is the `final_response` string the agent emitted when it finally declared
done (for the example task: *"Filtered Walmart careers results show 17 open
Support Services jobs in Bentonville, Arkansas."*).

Per-task run concurrency is 300; workers run as subprocesses so Browserbase
sessions and gateway calls happen in parallel. The gateway credentials and
Browserbase quota are the effective bottleneck.

---

## 4. Offline WebJudge evaluation (after the run)

Two independent grading layers exist, and they must not be confused:

### 4.1 In-agent `self_reflection` (gate, not final score)

`miniswewebagent.tools.self_reflection` runs *during* the task, using the
gpt-5.4 gateway on Stage 1 per-image scoring and Stage 2 aggregated verdict.
Its only role is to unblock the `done: true` gate. Its verdict is cached in
`final_runs/run_<id>/judge_result.json`.

### 4.2 External WebJudge (final reported score)

After (or after-the-fact for completed runs) the directory is scored with
the upstream Online-Mind2Web judge via
[scripts/eval_with_original_om2w.py](../scripts/eval_with_original_om2w.py).
That script:

1. Iterates every `<task_id>/` folder in
   `outputs/default/0421_all_best_default_json_sum20_s300_final_resp`.
2. Resolves the **latest** `final_runs/run_<id>/` per task using
   `run_<id>` numeric sorting.
3. Builds the judge input:
   - `task` from `result.json["task"]` / the benchmark `confirmed_task`.
   - `last_actions` from every `step <N> action: ...` line of
     `final_runs/run_<latest>/final_script_log.txt`.
   - `images_path` from every `final_execution_<N>_*.png` under
     `final_runs/run_<latest>/screenshots/`, sorted by the numeric prefix.
4. Feeds this into the upstream
   `WebJudge_Online_Mind2Web_eval` implementation at
   `/home/luyadong/sandbox/Online-Mind2Web/src/methods/webjudge_online_mind2web.py`,
   using the local `om2w_judge.utils.OpenaiEngine` with
   `--model o4-mini --score_threshold 3`. Each call produces a JSONL row
   with `{task_id, predicted_label, reason, ...}` appended to
   `WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json`.
5. The per-task WebJudge label (`1 = success`, `0 = failure`) defines the
   benchmark's reported success rate.

The pattern used downstream for this run was
`outputs/default/0421_all_best_default_json_sum20_s300_final_resp_eval/`
(or the step-budget snapshots `..._N<budget>_eval/` for
partial-trajectory ablations; see the `_N50 / _N100 / _N150 / _N200 / _N250`
series created by `scripts/materialize_budget_snapshots.py` +
`scripts/eval_with_original_om2w.py`).

An embedded `run.judge_script: om2w_judge/run.py` is wired via
`run_online_mind2web_judge` so that per-task judging can also happen inline
during the benchmark run, but the canonical reported numbers for this run
come from the offline pass against the **same** `om2w_judge` codebase.

---

## 5. End-to-end pipeline diagram

```text
best_default_judge_json.yaml
        │
        ▼
miniswewebagent.run.benchmarks.om2w   ──►  outputs/default/<run>/config_snapshot/
        │  (workers=300, task-level=all, om2w_260220.json)
        ▼
for each task_id (parallel, up to 300):
        │
        │   run_one() in run/mini.py
        │        ├── local_workspace env  (bash, 240s timeout, cred.sh sourced)
        │        ├── default agent        (step_limit=300, summary_every_n_steps=20,
        │        │                          require_self_reflection_success=true)
        │        └── phyagi model         (response_mode=json_schema, gpt-5.4)
        │
        │   loop:
        │     model → JSON {thought, bash_command, done, final_response}
        │     harness runs bash_command inside <task_id>/
        │       ├─ agent builds plan.md + judge_config.json
        │       ├─ agent drives Browserbase via heredoc Playwright scripts
        │       ├─ agent writes final_script.py + executes into final_runs/run_<id>/
        │       ├─ agent calls miniswewebagent.tools.self_reflection
        │       │     ├─ Stage 1: per-screenshot Score/Reasoning (parallel)
        │       │     └─ Stage 2: aggregated Status: success|failure
        │       └─ judge_result.json written per run_<id>
        │     done gate enforces predicted_label==1
        │
        ▼
per task_id/:
  task.json · plan.md · judge_config.json · trajectory.json · raw_responses.jsonl
  steps/ · screenshots/ · debug/steps/ · trajectory/ · final_runs/run_*/ · result.json
        │
        ▼
scripts/eval_with_original_om2w.py     (run separately, --model o4-mini --num_worker 300)
        │  pulls last_actions + images_path from the latest final_runs/run_<id>/
        │  calls upstream WebJudge_Online_Mind2Web_eval
        ▼
<run>_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json
```

---

## 6. Quick reference: names in the output directory

- `0421` — kickoff date.
- `all` — `--task-level all` (260 tasks).
- `best_default_judge_json` — config variant (default agent, JSON step
  contract, self-reflection judge gate).
- `sum20` — `agent.summary_every_n_steps: 20`.
- `s300` — `agent.step_limit: 300`.
- `final_resp` — this specific rerun where the config's emphasis was on
  returning a grounded `final_response` datum after the judge gate passed.
