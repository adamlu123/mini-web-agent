# Actor-Orchestrator 27-Task Result

## Decision

Reject broad promotion. Keep the pinned 55% harness as the default.

The complete merged panel improved the 14 treatment tasks by `+1.305159`, but
regressed the 13 protection tasks by `-0.400000`:

| Group | Tasks | Baseline | Orchestrator | Delta |
| --- | ---: | ---: | ---: | ---: |
| Treatments | 14 | 4.696081 | 6.001240 | +1.305159 |
| Protections | 13 | 13.000000 | 12.600000 | -0.400000 |
| Overall | 27 | 17.696081 | 18.601240 | +0.905159 |

Mean overall delta was `+0.033524` per task. Protection regressions on `039`
and `100` violate the predeclared protection-first promotion rule.

## Provenance

- Branch: `cli-harness-v2-actor-orchestrator`
- Actor-orchestrator behavior: `590cf427`
- AIServices routing commit: `e727ca43`
- Model: GPT-5.5 xhigh
- Budget: 300 shared model calls per task
- Compaction interval: 40 Actor tool steps
- Release: `v2026.06.24`
- Baseline: exact selected attempts from the pinned 106-task result

Primary width-27 root:

```text
results-actor-orchestrator27-aiservices-w27-20260818T201204Z
```

It evaluated 24 tasks. Task `088` failed guest command infrastructure; tasks
`100` and `102` received upstream `invalid_prompt` rejections. These are not
behavioral zeros.

The identical low-width retries completed all three:

```text
results-actor-orchestrator27-aiservices-retry3-w3-20260818T230645Z
```

The first non-AIServices width-27 attempt is invalid. It exhausted gateway
retries on 14 tasks. Adding:

```json
{
  "strict_session": true,
  "infrastructure": "aiservices"
}
```

eliminated HTTP 429s in the primary rerun.

## Per-task scores

| Task | Group | Baseline | Orchestrator | Delta |
| --- | --- | ---: | ---: | ---: |
| 001 | protection | 1.000000 | 1.000000 | +0.000000 |
| 003 | protection | 1.000000 | 1.000000 | +0.000000 |
| 006 | protection | 1.000000 | 1.000000 | +0.000000 |
| 008 | treatment | 0.670000 | 0.835000 | +0.165000 |
| 012 | protection | 1.000000 | 1.000000 | +0.000000 |
| 013 | protection | 1.000000 | 1.000000 | +0.000000 |
| 020 | treatment | 0.100000 | 0.100000 | +0.000000 |
| 022 | protection | 1.000000 | 1.000000 | +0.000000 |
| 025 | treatment | 0.724100 | 0.812188 | +0.088089 |
| 028 | protection | 1.000000 | 1.000000 | +0.000000 |
| 030 | protection | 1.000000 | 1.000000 | +0.000000 |
| 033 | treatment | 0.000000 | 0.000000 | +0.000000 |
| 036 | treatment | 0.104000 | 0.710989 | +0.606989 |
| 039 | protection | 1.000000 | 0.800000 | -0.200000 |
| 049 | treatment | 0.200000 | 0.200000 | +0.000000 |
| 055 | treatment | 0.000000 | 0.470000 | +0.470000 |
| 057 | treatment | 0.653400 | 0.655500 | +0.002100 |
| 062 | treatment | 0.543637 | 0.218278 | -0.325359 |
| 065 | protection | 1.000000 | 1.000000 | +0.000000 |
| 073 | protection | 1.000000 | 1.000000 | +0.000000 |
| 081 | treatment | 0.000000 | 0.700000 | +0.700000 |
| 088 | treatment | 0.230000 | 0.230000 | +0.000000 |
| 095 | protection | 1.000000 | 1.000000 | +0.000000 |
| 098 | treatment | 0.703704 | 0.629630 | -0.074074 |
| 100 | protection | 1.000000 | 0.800000 | -0.200000 |
| 102 | treatment | 0.100000 | 0.000000 | -0.100000 |
| 108 | treatment | 0.667241 | 0.439655 | -0.227586 |

## What caused the gains

The visible shared step counter was not the primary mechanism. Only a minority
of the strongest gains reached the midpoint reminder, and tasks with similar
Actor budgets and compaction counts had opposite outcomes.

The strongest repeated mechanism was:

```text
Actor owns orchestration
-> specialist report is read before execution/submission
-> Verifier inspects a saved candidate before submission
-> original Actor repairs direct findings
-> Actor optionally re-verifies
-> submit
```

### Direct examples

- `008 (+0.165)`: three Verifier cycles caught reversed dates, stale canonical
  state, and attachment uncertainty. Actor repaired and reverified before
  submission.
- `081 (+0.700)`: Verifier found a missing citation key; Actor corrected it and
  a second Verifier confirmed the repaired live service/project.
- `036 (+0.606989)`: adaptive second Binder/Critic consultation resolved exact
  Zotero scope and preservation decisions before submission.

The step counter and shared budget were useful enablers because the Actor could
decide whether to spend calls on execution or feedback, but no trace isolates
budget narration itself as causal.

### Gains not attributable to orchestration

- `055 (+0.47)` retained the reference MP4 while its editable Shotcut project
  remained wrong. This repeats the previous artifact-shortcut result and does
  not validate the orchestration mechanism.
- `025 (+0.088089)` used essentially the same native PDF-writing method as the
  baseline; evaluator variance and small repair differences remain plausible.

## Why broad promotion fails

- `039 (1 -> .8)`: Actor consumed the full 300-call budget, left only two calls
  for Verifier, and submitted incomplete late-state moderation coverage.
- `100 (1 -> .8)`: Actor spent 285 calls in screenshot-driven trial and error,
  never called Verifier, and was forced to submit with four disconnected puzzle
  groups.
- `102 (.1 -> 0)`: Actor knowingly treated image-only DOCX files as adequate;
  Critic warned about editable reconstruction, but Actor did not enforce it and
  Verifier accepted structural validity without checking substantial editable
  text.
- `062 (.543637 -> .218278)`: a `ready` Verifier report did not imply correct
  evaluator-facing Zotero state.

Free-form control therefore improves some tasks through timely consultation,
but can also let the Actor starve verification, ignore correct specialist
advice, or accept weak verification.

## Aggregate orchestration behavior

Across the 23 primary-run scored tasks available during the causal audit:

- Actor calls: 1,618
- Binder calls: 313
- Critic calls: 279
- Verifier calls: 440
- Binder/Critic/Verifier invocations: 24/25/33
- Compactions: 28 across 15 tasks
- Midpoint reminders: four tasks
- Finalization reminders: one task

Repeated Verifier calls often produced useful repairs, but Verifier quality and
budget allocation were inconsistent.

## Conclusion

The main-Actor/agents-as-tools architecture is a promising mechanism-level
result, not a promotable harness. Future refinement should preserve pre-submit
Verifier feedback while adding protection constraints:

- reserve a minimum Verifier budget that Actor cannot spend;
- prevent submission without a substantive Verifier inspection;
- make Verifier test the task's native semantic requirement, not only artifact
  validity;
- add a bounded stop/replan rule before Actor exhausts the budget on a falsified
  visual strategy.

Do not promote or run a broader paid panel until those constraints pass a
repeated protection-first test.
