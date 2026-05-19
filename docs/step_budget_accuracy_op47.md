# Step-Budget Accuracy Report — Online-Mind2Web (260220, N=300) — Claude Opus 4.7

**Judge:** upstream `WebJudge_Online_Mind2Web_eval` (from `Online-Mind2Web`), model `o4-mini`, score threshold `3`
**Benchmark:** `om2w_260220.json` — 300 tasks (80 easy / 143 medium / 77 hard)
**Runs:** mini-web-agent `0422_all_best_default_json_sum20_s300_final_resp_op47` (config `best_default_judge_json_claude.yaml`, model `claude-opus-4-7`) with step budgets capped at N=50, 100, 150, 200, 250, 300
**Generated:** 2026-04-23

## Accuracy over full benchmark (denominator = benchmark totals)

Tasks with no agent trajectory are counted as failures. This is the canonical success rate.

| Step Budget | Easy (/80)          | Medium (/143)         | Hard (/77)           | Overall (/300)          |
| ----------- | ------------------- | --------------------- | -------------------- | ----------------------- |
| 50          | 72 / 80 = 90.00%    | 117 / 143 = 81.82%    | 59 / 77 = 76.62%     | 248 / 300 = **82.67%**  |
| 100         | 74 / 80 = 92.50%    | 118 / 143 = 82.52%    | 62 / 77 = 80.52%     | 254 / 300 = **84.67%**  |
| 150         | 77 / 80 = 96.25%    | 121 / 143 = 84.62%    | 59 / 77 = 76.62%     | 257 / 300 = **85.67%**  |
| 200         | 71 / 80 = 88.75%    | 123 / 143 = 86.01%    | 59 / 77 = 76.62%     | 253 / 300 = **84.33%**  |
| 250         | 74 / 80 = 92.50%    | 120 / 143 = 83.92%    | 57 / 77 = 74.03%     | 251 / 300 = **83.67%**  |
| 300         | 74 / 80 = 92.50%    | 115 / 143 = 80.42%    | 60 / 77 = 77.92%     | 249 / 300 = **83.00%**  |

Peak overall accuracy at **N=150 (85.67%)**.

## Accuracy over evaluated tasks only (denominator = tasks with trajectory)

Excludes tasks whose agent trajectories were missing or had no screenshots.

| Step Budget | Easy               | Medium              | Hard               | Overall              | Rows |
| ----------- | ------------------ | ------------------- | ------------------ | -------------------- | ---- |
| 50          | 72 / 79 = 91.14%   | 117 / 140 = 83.57%  | 59 / 73 = 80.82%   | 248 / 292 = 84.93%   | 292  |
| 100         | 74 / 80 = 92.50%   | 118 / 140 = 84.29%  | 62 / 73 = 84.93%   | 254 / 293 = 86.69%   | 293  |
| 150         | 77 / 80 = 96.25%   | 121 / 140 = 86.43%  | 59 / 73 = 80.82%   | 257 / 293 = 87.71%   | 293  |
| 200         | 71 / 80 = 88.75%   | 123 / 140 = 87.86%  | 59 / 73 = 80.82%   | 253 / 293 = 86.35%   | 293  |
| 250         | 74 / 80 = 92.50%   | 120 / 140 = 85.71%  | 57 / 73 = 78.08%   | 251 / 293 = 85.67%   | 293  |
| 300         | 74 / 80 = 92.50%   | 115 / 140 = 82.14%  | 60 / 73 = 82.19%   | 249 / 293 = 84.98%   | 293  |

## Missing tasks

Seven benchmark tasks never produced a usable trajectory for any step budget in this run.
All 7 aborted mid-run with the same Anthropic API 400 error at ~2026-04-23T02:00 UTC:
```
{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}
```

| Task ID                                      | Level  | In N=50 snapshot? | In N≥100 snapshots? |
| -------------------------------------------- | ------ | ----------------- | ------------------- |
| `1ab384fb3a791edfb410213cc6b82151_110325`    | hard   | ✗                 | ✗                   |
| `2c8ef01a92c71ba9ef2e59bb17eea2b3_110325`    | medium | ✗                 | ✗                   |
| `7e6993f2c5cd72c44809024f0bc85dc1`           | medium | ✗                 | ✗                   |
| `9829f3087ab1f9c8eba6b6dd2b831d25_112325`    | easy   | ✗                 | ✓ (6 screenshots)*  |
| `a48e2f1ee8d87eaeea56fe5e730427e6`           | medium | ✗                 | ✗                   |
| `c801d1c951f59297f526bab84fa86c6e`           | hard   | ✗                 | ✗                   |
| `fc53ddd3421411a41c1020a3fdc84ec4`           | hard   | ✗                 | ✗                   |

\* `9829f3087ab1f9c8eba6b6dd2b831d25_112325` did produce 6 screenshots under `final_runs/run_001` before failing, so it is included in N=100..300 snapshots but still has no screenshots at N=50 (the 6 shots are beyond step 50's cutoff window, per snapshot construction).

A retry of these 7 tasks was launched in screen session `op47_retry7` into `outputs/default/0423_op47_retry7/`, but immediately failed again with the same credit-balance error — rerun after billing is restored.

## Notes

- N=50 snapshot has 293 task dirs (300 − 7 missing). Eval skipped 1 (`9829f3087ab1f9c8eba6b6dd2b831d25_112325` has dir but 0 PNGs at N=50), yielding 292 rows.
- N=100..300 snapshots each have 294 task dirs (300 − 6 missing). Eval skipped 1 (`949dc965a6c23a95663b3bc2ca2c3a8a` — 0 PNGs under its `run_001/screenshots/`), yielding 293 rows.
- Eval script: [scripts/eval_with_original_om2w.py](../scripts/eval_with_original_om2w.py) with `--plain_text` action loader, `--num_worker 300`.
  - The `load_screenshots` helper was relaxed in this session to accept any `*.png` (not only `final_execution_*`) so it handles op47's naming (`01_*`, `cp1_*`, `critical_point_*`).
- Raw per-task results:
  - [0422_all_best_default_json_sum20_s300_final_resp_op47_N50_eval](../outputs/default/0422_all_best_default_json_sum20_s300_final_resp_op47_N50_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0422_all_best_default_json_sum20_s300_final_resp_op47_N100_eval](../outputs/default/0422_all_best_default_json_sum20_s300_final_resp_op47_N100_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0422_all_best_default_json_sum20_s300_final_resp_op47_N150_eval](../outputs/default/0422_all_best_default_json_sum20_s300_final_resp_op47_N150_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0422_all_best_default_json_sum20_s300_final_resp_op47_N200_eval](../outputs/default/0422_all_best_default_json_sum20_s300_final_resp_op47_N200_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0422_all_best_default_json_sum20_s300_final_resp_op47_N250_eval](../outputs/default/0422_all_best_default_json_sum20_s300_final_resp_op47_N250_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0422_all_best_default_json_sum20_s300_final_resp_op47_N300_eval](../outputs/default/0422_all_best_default_json_sum20_s300_final_resp_op47_N300_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)

## Comparison vs gpt-5.4 (same benchmark, same judge)

| Step Budget | gpt-5.4 Overall | claude-opus-4-7 Overall |
| ----------- | --------------: | ----------------------: |
| 50          |      **82.00%** |              **82.67%** |
| 100         |      **86.67%** |              **84.67%** |
| 150         |      **83.67%** |              **85.67%** |
| 200         |      **83.33%** |              **84.33%** |
| 250         |      **84.67%** |              **83.67%** |
| 300         |      **85.33%** |              **83.00%** |

See [step_budget_accuracy_gpt54.md](step_budget_accuracy_gpt54.md) for the gpt-5.4 report.
