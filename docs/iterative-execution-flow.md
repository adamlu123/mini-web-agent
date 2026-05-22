# Iterative Execution Flow

The iterative runner wraps the single-shot flow in [execution-flow.md](execution-flow.md) in an outer loop: each round runs a fresh agent, collects artifacts, asks a judge, and either returns or resets and retries.

## 1. Pipeline

```text
+----------------------------------------------------------------------------+
| Input                                                                      |
|   CLI: mini-web (with iterative_judge.yaml) or run/benchmarks/om2w         |
|   Config: iterative.max_rounds, judge_mode, round1/next_round templates    |
|   Code: run/mini.py, agents/iterative.py, config/iterative_judge.yaml      |
+----------------------------------------------------------------------------+
                                    |
                                    v
+----------------------------------------------------------------------------+
| Stage A: Runner bootstrap                                                  |
|   Build model + env + IterativeRunner (replaces DefaultAgent at top level) |
|   Resolve task, start_url, output_dir; prepare browser once                |
|   Code: run/mini.py, agents/iterative.py (IterativeRunner.__init__)        |
|   Output: live model/env, prev_artifacts=None                              |
+----------------------------------------------------------------------------+
                                    |
                                    v
+============================================================================+
| Per-round loop  (round_id = 1 .. max_rounds)                               |
+============================================================================+
     |
     v
+----------------------------------------------------------------------------+
| Stage B: Render round instance message                                     |
|   round==1 -> round1_instance_template                                     |
|   round>1  -> next_round_instance_template + attached prev screenshots     |
|   Vars: env.get_template_vars(), model.get_template_vars(),                |
|         task, task_id, start_url, round_id,                                |
|         prev_round_dir, prev_final_script, prev_command_output,            |
|         prev_judge_response                                                |
|   Jinja2 StrictUndefined; literal braces need {% raw %}...{% endraw %}     |
+----------------------------------------------------------------------------+
     |
     v
+----------------------------------------------------------------------------+
| Stage C: Inner agent round                                                 |
|   _IterativeInnerAgent: fresh messages=[], n_calls=0, step_limit=50        |
|   Runs the Stage D main loop from execution-flow.md in the SAME browser    |
|   Ends when agent emits done=true or hits step_limit                       |
|   Prompt flow: agent uses bash to run two_stage_judge tool, writes         |
|   final_runs/run_N/{final_script.py, judge_result.json, screenshots/}      |
|   Code: agents/iterative.py (_IterativeInnerAgent), agents/default.py      |
|   Files: <output_dir>/round_<N>/{trajectory.json, debug/, screenshots/}    |
+----------------------------------------------------------------------------+
     |
     v
+----------------------------------------------------------------------------+
| Stage D: Completion gate — _collect_round_artifacts                        |
|   - pick latest workspace/final_runs/run_N/ dir                            |
|   - copy workspace/final_script.py into run_N/                             |
|   - extract command_output from debug/steps/step_*.json                    |
|   - collect run_N/screenshots/final_execution_*.png                        |
|     (fallback: workspace/screenshots/step_*.png)                           |
|   Gate fail (no run_N/) -> skip judge, prev_artifacts=None, next round     |
|   Code: agents/iterative.py (_collect_round_artifacts,                     |
|         _resolve_latest_final_run_dir)                                     |
+----------------------------------------------------------------------------+
     |
     v
+----------------------------------------------------------------------------+
| Stage E: Judge dispatch (judge_mode)                                       |
|   om2w       -> _run_judge (legacy WebJudge LLM call, inline)              |
|   workspace  -> _run_workspace_judge (exec agent-authored judge.py)        |
|   tool       -> _read_tool_judge_result: read run_N/judge_result.json      |
|                 written by the agent's two_stage_judge CLI call            |
|   Result: JudgeResult(success, response_text, record)                      |
|   Persisted as round_dir/iterative_judge_result.json (runner's own copy;   |
|   does NOT clobber the agent's judge_result.json)                          |
|   Code: agents/iterative.py, tools/two_stage_judge.py                      |
+----------------------------------------------------------------------------+
     |
     v
+----------------------------------------------------------------------------+
| Stage F: Verdict routing                                                   |
|   success -> last_result._iterative_success=True,                          |
|              _iterative_rounds=round_id, _final_round_dir=run_N,           |
|              return (browser stays live for caller to close)               |
|   failure -> prev_artifacts = round_artifacts;                             |
|              next round resets agent context (messages=[], n_calls=0)      |
|              but reuses the same browser session/page                      |
+----------------------------------------------------------------------------+
     |
     +---- max_rounds exhausted -> return with _iterative_success=False
     v
+----------------------------------------------------------------------------+
| Stage G: Finalization                                                      |
|   Same as execution-flow.md Stage E: close env, write result.json,         |
|   normalize trajectory/ screenshots                                        |
|   Extra keys in result: _iterative_success, _iterative_rounds,             |
|                         _final_round_dir                                   |
+----------------------------------------------------------------------------+
```

## 2. What each round writes

```text
<output_dir>/
  round_1/
    trajectory.json
    debug/steps/step_*.json
    screenshots/step_*.png
  round_2/
    ...
  result.json                   # summary across all rounds

<workspace_dir>/                # shared across rounds (same browser session)
  final_script.py               # overwritten each round
  screenshots/step_*.png
  final_runs/
    run_001/
      final_script.py
      command_output.txt
      judge_result.json          # agent-produced (two_stage_judge output)
      iterative_judge_result.json# runner's copy of JudgeResult
      screenshots/
        final_execution_*.png
    run_002/ ...                 # new run dir created by agent each round
```

## 3. Judge modes

| mode        | how verdict is produced                                               | agent responsibility                         |
|-------------|------------------------------------------------------------------------|----------------------------------------------|
| `om2w`      | runner calls WebJudge LLM directly using run_N artifacts               | just produce final_runs/run_N/               |
| `workspace` | runner execs `workspace/judge.py` authored by the agent                | write a passing judge.py                     |
| `tool`      | runner reads `run_N/judge_result.json` written by `two_stage_judge`    | call the `two_stage_judge` CLI once per round|

`tool` is the current default in `config/iterative_judge.yaml`.

## 4. Mental model

```text
iterative runner
  -> for each round:
       render round template (round1 or next_round + prev artifacts)
       -> run inner DefaultAgent loop until done
       -> locate final_runs/run_N/ (completion gate)
       -> get verdict (tool reads judge_result.json)
       -> success? return.  failure? reset context, keep browser, retry.
  -> after max_rounds, return with _iterative_success=False
```

The single-shot loop in [execution-flow.md](execution-flow.md) is unchanged; iterative only adds the outer round/judge/reset layer.
