# 0418_all_best_repro — Eval Analysis

Analysis of judge results for `outputs/iterative/0418_all_best_repro` (300 tasks, OM2W 260220).

- Judge run: `outputs/iterative/0418_all_best_repro_eval_20260419_233925`
- Judge: `WebJudge_Online_Mind2Web_Sandbox_eval` / model `o4-mini` / score_threshold=3
- Source summary: [outputs/iterative/0418_all_best_repro_eval_20260419_233925/summary.md](../outputs/iterative/0418_all_best_repro_eval_20260419_233925/summary.md)

## Overall

| metric | value |
|---|---:|
| total task dirs | 301 (1 non-task `config_snapshot` skipped) |
| judged records | 300 |
| success (predicted_label=1) | 213 |
| failure (predicted_label=0) | 87 |
| success rate (judged) | **71.00%** |
| success rate (all 301) | **70.76%** |

## Success rate by task level

Levels are taken from `src/miniswewebagent/tasks/om2w_260220.json`.

| level | total | success | success rate |
|---|---:|---:|---:|
| easy | 80 | 66 | **82.50%** |
| medium | 143 | 101 | **70.63%** |
| hard | 77 | 46 | **59.74%** |
| **all** | **300** | **213** | **71.00%** |

Monotonic drop with difficulty (~23 pp gap between easy and hard).

## Success rate by agent exit_status

| exit_status | judged | success | success rate |
|---|---:|---:|---:|
| Submitted | 251 | 200 | 79.68% |
| HTTPStatusError | 23 | 9 | 39.13% |
| LimitsExceeded | 26 | 4 | 15.38% |

A `Submitted` trajectory is ~5x more likely to be judged correct than a `LimitsExceeded` one.

## Main failure modes (87 judged failures)

Categories are mutually exclusive, assigned by keyword classification on the judge's `evaluation_details.response`.

| category | count | share |
|---|---:|---:|
| filters_not_applied_or_partial | 36 | 41% |
| agent_step_limit_exceeded (`LimitsExceeded`) | 22 | 25% |
| agent_http_error (`HTTPStatusError`) | 14 | 16% |
| answer_missing_in_final_response | 7 | 8% |
| other (semantic mismatch / wrong scope) | 5 | 6% |
| task_partially_completed | 2 | 2% |
| wrong_result_returned | 1 | 1% |

### Examples per category

**filters_not_applied_or_partial** (36) — agent reached the right page but did not apply (or only partially applied) the requested filters/refinements.
- `[medium/Submitted] 4639a54f3ab549864fd8d60b7398b1e1` — *Find a white female kitten within 35 miles of zip 77494.* Filters were applied, but the final snapshot is obscured so the judge could not verify a matching listing.
- `[hard/Submitted] fc53ddd3421411a41c1020a3fdc84ec4` — *Open-box Galaxy S25 Plus + trade-in gray S20 5G (Verizon)…* Got the open-box price but never produced a trade-in value.

**agent_step_limit_exceeded** (22) — agent ran out of steps before producing a usable answer.
- `[hard/LimitsExceeded] 949dc965a6c23a95663b3bc2ca2c3a8a` — *UA/AA flights LHR→JFK arriving 8–11 PM on FlightAware.* Action history contains only setup/logging, no FlightAware navigation.
- `[easy/LimitsExceeded] bb518416a786fdb9b9bbf0c78515595e` — *Browse the class schedule of graduate-level CS courses.* Never reached a schedule page.

**agent_http_error** (14) — trajectory aborted by HTTPStatusError (typically gateway/site error). Still 39% pass rate because the agent often submitted a partial answer first.
- `[easy/HTTPStatusError] 6db4a0e346976f2729ba9afcd3208941` — *Look up tracking for shipment #3023858502.* Never entered the number.
- `[easy/HTTPStatusError] a6f0434ce6aff5f9b03681241b03ad82` — *Tesla closing price on 2023-03-17.* Agent got `$180.13` but didn't surface it in the final response.

**answer_missing_in_final_response** (7) — task essentially solved, but the required value/details weren't included in `final_result_response`.
- `[hard/Submitted] 905cb53061c33aa2d77e485fe1fca516` — *Dermatologists within 10 mi of 10019 accepting Blue Medicare Advantage.* Searched but didn't apply the distance/insurance filters in the recorded steps.
- `[medium/Submitted] 64b76158720a69e4a5c31a55d54928bf` — *Compare two pescatarian diets.* Compared pescatarian vs vegetarian instead of two pescatarian variants.

**other / semantic mismatch** (5)
- `[easy/Submitted] 853afd530c72f4b00ffc32ae854efaf8` — *Wind flow map for Belo Horizonte.* Showed Brazil-scale map, not Belo Horizonte.
- `[medium/Submitted] 239a29bde438fe44fe17fe1390ef1634` — *Gluten-free diet to lose weight for a pregnant woman.* Returned two unrelated articles instead of a combined plan.

**task_partially_completed** (2)
- `[medium/Submitted] b3a7da968de13bbdcaed12ffe4993df6` — *Compare Afghan Hound, Akita, Azawakh.* Selected the breeds but never executed the comparison.

**wrong_result_returned** (1)
- `[easy/Submitted] 871e7771cecb989972f138ecc373107b` — *Vancouver BC 7-day weather.* Reached the right page but reported temperatures that don't match the page.

## Takeaways

- **Filter-application failures dominate (~41% of all failures).** The agent often reaches the correct site/results page but misses one or more filter steps, especially on multi-constraint queries (insurance type, distance, color/sex/age combinations).
- **Step-budget exhaustion is the second-largest bucket (~25%)**, heavily concentrated in `hard` tasks (10 of 22 `LimitsExceeded` are hard).
- **HTTPStatusError aborts** account for ~16%; ~39% of those still pass because the agent had already produced a usable answer before the abort.
- A small but distinct **"answered correctly but didn't surface it"** bucket (~8%) suggests gains are possible from a stricter final-response template.

## Reproducing the analysis

```python
import json, collections
EVAL = "outputs/iterative/0418_all_best_repro_eval_20260419_233925"
recs = [json.loads(l) for l in open(f"{EVAL}/WebJudge_Online_Mind2Web_Sandbox_eval_o4-mini_score_threshold_3_auto_eval_results.json")]
tasks = {t["task_id"]: t for t in json.load(open("src/miniswewebagent/tasks/om2w_260220.json"))}

# By level
by_level = collections.defaultdict(lambda: [0, 0])
for r in recs:
    lv = tasks[r["task_id"]]["level"]
    by_level[lv][0] += 1
    by_level[lv][1] += int(r.get("predicted_label") == 1)

# By exit_status
by_exit = collections.defaultdict(lambda: [0, 0])
for r in recs:
    es = r.get("exit_status") or "unknown"
    by_exit[es][0] += 1
    by_exit[es][1] += int(r.get("predicted_label") == 1)
```
