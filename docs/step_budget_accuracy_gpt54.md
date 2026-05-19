# Step-Budget Accuracy Report — Online-Mind2Web (260220, N=300)

**Judge:** upstream `WebJudge_Online_Mind2Web_eval` (from `Online-Mind2Web`), model `o4-mini`, score threshold `3`
**Benchmark:** `om2w_260220.json` — 300 tasks (80 easy / 143 medium / 77 hard)
**Runs:** mini-web-agent `0421_all_best_default_json_sum20_s300` with step budgets capped at N=50, 100, 150, 200, 250, 300
**Generated:** 2026-04-22

## Accuracy over full benchmark (denominator = benchmark totals)

Tasks with no agent trajectory are counted as failures. This is the canonical success rate.

| Step Budget | Easy (/80)          | Medium (/143)         | Hard (/77)           | Overall (/300)        |
| ----------- | ------------------- | --------------------- | -------------------- | --------------------- |
| 50          | 77 / 80 = 96.25%    | 116 / 143 = 81.12%    | 53 / 77 = 68.83%     | 246 / 300 = **82.00%** |
| 100         | 75 / 80 = 93.75%    | 126 / 143 = 88.11%    | 59 / 77 = 76.62%     | 260 / 300 = **86.67%** |
| 150         | 74 / 80 = 92.50%    | 118 / 143 = 82.52%    | 59 / 77 = 76.62%     | 251 / 300 = **83.67%** |
| 200         | 72 / 80 = 90.00%    | 122 / 143 = 85.31%    | 56 / 77 = 72.73%     | 250 / 300 = **83.33%** |
| 250         | 73 / 80 = 91.25%    | 125 / 143 = 87.41%    | 56 / 77 = 72.73%     | 254 / 300 = **84.67%** |
| 300         | 77 / 80 = 96.25%    | 123 / 143 = 86.01%    | 56 / 77 = 72.73%     | 256 / 300 = **85.33%** |

## Accuracy over evaluated tasks only (denominator = tasks with trajectory)

Excludes tasks whose agent trajectories were missing from the run directory.

| Step Budget | Easy              | Medium             | Hard              | Overall            | Rows |
| ----------- | ----------------- | ------------------ | ----------------- | ------------------ | ---- |
| 50          | 77 / 80 = 96.25%  | 116 / 143 = 81.12% | 53 / 76 = 69.74%  | 246 / 299 = 82.27% | 299  |
| 100         | 75 / 80 = 93.75%  | 126 / 143 = 88.11% | 59 / 76 = 77.63%  | 260 / 299 = 86.96% | 299  |
| 150         | 74 / 80 = 92.50%  | 118 / 143 = 82.52% | 59 / 76 = 77.63%  | 251 / 299 = 83.95% | 299  |
| 200         | 72 / 80 = 90.00%  | 122 / 143 = 85.31% | 56 / 76 = 73.68%  | 250 / 299 = 83.61% | 299  |
| 250         | 73 / 80 = 91.25%  | 125 / 143 = 87.41% | 56 / 76 = 73.68%  | 254 / 299 = 84.95% | 299  |
| 300         | 77 / 79 = 97.47%  | 123 / 143 = 86.01% | 56 / 76 = 73.68%  | 256 / 298 = 85.91% | 298  |

## Notes

- N=50..250 each have 299 / 300 tasks with trajectories; the one missing (in every run) is hard task `c03ee2be3d73556ab789c0ad1cbd3451` (AKC dog-groomer search).
- N=300 has 298 rows: 299 materialized task folders minus 1 skip (`bf3b311cc8dce16d3de844f4b5875dfd` has no screenshots under its latest `run_*`). Plus the same missing benchmark id as above.
- The eval script used is [scripts/eval_with_original_om2w.py](scripts/eval_with_original_om2w.py) with `--plain_text` action loader, 300 parallel workers.
- Raw per-task results:
  - [0421_all_best_default_json_sum20_s300_N50_eval](outputs/default/0421_all_best_default_json_sum20_s300_N50_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0421_all_best_default_json_sum20_s300_N100_eval](outputs/default/0421_all_best_default_json_sum20_s300_N100_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0421_all_best_default_json_sum20_s300_N150_eval](outputs/default/0421_all_best_default_json_sum20_s300_N150_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0421_all_best_default_json_sum20_s300_N200_eval](outputs/default/0421_all_best_default_json_sum20_s300_N200_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0421_all_best_default_json_sum20_s300_N250_eval](outputs/default/0421_all_best_default_json_sum20_s300_N250_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)
  - [0421_all_best_default_json_sum20_s300_N300_eval](outputs/default/0421_all_best_default_json_sum20_s300_N300_eval/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json)

## Token usage & cost per step budget

Cumulative `input_tokens` / `output_tokens` / `cached_input_tokens` are read from each task's `debug/steps.md` (the `cumulative_request` / `cumulative_response` JSON block written at every step). For a budget N we take the largest step ≤ N that has usage recorded. All 300 tasks report usage.

**Pricing (gpt-5.4, short context, per 1M tokens):** input $2.50, cached input $0.25, output $15.00.
Cost formula: `(input − cached) · $2.50 + cached · $0.25 + output · $15.00`, all per 1M tokens.

| Step Budget | Sum input tokens | Sum cached input | Sum output tokens | Avg input | Avg output | **Total cost (USD)** | **Avg cost / task (USD)** |
| ----------- | ---------------: | ---------------: | ----------------: | --------: | ---------: | -------------------: | ------------------------: |
| 50          |      246,030,307 |       15,031,552 |         8,700,613 |   820,101 |     29,002 |           **711.76** |                **2.3725** |
| 100         |      336,983,440 |       21,874,560 |        10,764,295 | 1,123,278 |     35,881 |           **954.71** |                **3.1824** |
| 150         |      390,681,701 |       27,458,944 |        12,041,165 | 1,302,272 |     40,137 |         **1,095.54** |                **3.6518** |
| 200         |      416,110,929 |       30,831,104 |        12,570,206 | 1,387,036 |     41,901 |         **1,159.46** |                **3.8649** |
| 250         |      436,899,060 |       34,420,480 |        13,064,434 | 1,456,330 |     43,548 |         **1,210.77** |                **4.0359** |
| 300         |      446,738,012 |       36,241,792 |        13,312,053 | 1,489,127 |     44,374 |         **1,234.98** |                **4.1166** |

- Script: [scripts/token_usage_by_step_budget.py](scripts/token_usage_by_step_budget.py)
- Per-task CSV: `/tmp/token_usage_by_budget.csv`
