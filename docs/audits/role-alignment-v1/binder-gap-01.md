# Binder exact-value gaps: tasks 007, 010, 031, 034

| Task | Score | Primary cause | Actor relation |
|---|---:|---|---|
| 007 | .60 | wrong workbook cell contract | compliant |
| 010 | 0 | paragraph-only evaluator conflict | no material Actor failure |
| 031 | 5/7 | two omitted hidden violations | compliant |
| 034 | .59375 | underbound task-visible exact values | followed binding, violated task facts |

## Task 007

Final binding required `[0,0,0,0,2,2]`; evaluator gold was
`[0,1,3,0,2,2]`. Critic corrected thread/reply/filename requirements but did
not independently resolve the two wrong cells. Actor faithfully wrote the wrong
values while completing the mail task.

## Task 010

No wrong table value is evidenced. The evaluator ignores tables and performs an
all-or-nothing exact paragraph comparison. One unavailable paragraph mismatch
can zero the result. Critic’s visible corrections reached the artifact.

## Task 031

Final binding omitted:

- Team 1 incomplete paper title
- Team 3 incomplete author list

Critic accepted the draft unchanged. Actor posted exactly the final binding,
yielding 5/7 under the hidden evaluator list.

## Task 034

Task-visible canonical values were:

- `projectStartDate = 2025-07-01`
- `mansonmh@hku.hk`
- `budgetCap = 60000.00`

Final binding weakened or retained incorrect versions, and Actor submitted
`2025-09-01`, `mansonmh@hku.hku`, and `48000.00`.

Critic improved reasons/justifications and row cardinality but failed to elevate
the task-provided exact date, corrected email, and cap.

## Prevention principles

1. Task-provided corrections outrank noisy drafts.
2. Promote exact literals to equality constraints; do not weaken them to
   “valid,” “positive,” or “contains.”
3. Independently recompute exact rows/cells where possible.
4. Critic must examine full leaf/list coverage, not only semantic summaries.
5. Actor compliance cannot repair a wrong binding.
6. Distinguish the artifact’s actual extraction surface, such as paragraph-only
   versus table-aware comparison.

## Evidence

- `reliability64-alignment-audit/task-007.md`
- `reliability64-alignment-audit/task-010.md`
- `reliability64-alignment-audit/task-031.md`
- `reliability64-alignment-audit/task-034.md`

