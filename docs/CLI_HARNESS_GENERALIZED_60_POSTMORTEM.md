# Generalized 60-Task Harness Postmortem

## Decision

The generalized profile is rejected as a development base.

- Preserve branch `cli-harness-v2-generalized-next` and tag
  `cli-harness-v2-generalized-60-w54-v1` only as historical evidence.
- Base future work on `master`, whose harness behavior remains `b091c78`.
- Use `cli-harness-v2-55pct-106-repro-v7` for exact baseline reproduction.
- Do not cherry-pick the generalized package wholesale.

The 60-task result improved only `+0.0559759381` total, or
`+0.0009329323/task`, while evaluated coverage fell from `57/60` to `53/60`.
The apparent gain was a cancellation of favorable trajectories,
infrastructure recovery, exact evaluator-format hits, and direct regressions.

## Why the package looked promising

The design targeted failures that genuinely appeared in baseline traces:

| Proposed mechanism | Reason for optimism |
|---|---|
| Critic probe disposition | Unresolved exact facts could disappear during Binder adjudication. |
| Compact manifests | Finite repeated sets needed explicit cardinality and identity coverage. |
| Per-proof Verifier provenance | Broad all-pass reports could lack direct evidence for individual proof clauses. |
| Completion routing | A small repair round was inappropriate when the main artifact was absent. |
| Local backup and rollback | Verifier-directed mutation needed a reversible safety boundary. |
| Native/accessibility-first Actor guidance | Several failures used fragile interaction or failed to persist canonical native state. |
| Real boot-disk expansion | The requested 80G disk previously did not enlarge the boot filesystem. |
| Root/service failure classification | Infrastructure failures were being charged to the Actor or evaluator. |

Each idea was individually plausible. The mistake was treating plausible
diagnostic mechanisms as a coherent score-improving package before proving
their activation, safety, and timing independently.

## What actually happened

### Direct harness regressions

- `100: 1.0 -> 0.0`: the Actor solved the puzzle with the anti-DevTools flag
  false. The Verifier attached through CDP and reloaded the judged session,
  causing the persistent flag to become true. The evaluator then hard-failed
  the Verifier-contaminated state.
- `095: 1.0 -> 0.0`: both runs began without required media. The baseline
  Actor used the legitimate ask-user-once recovery path. The generalized Actor
  treated the unresolved prerequisite as terminal and submitted no artifact.
- `042`: a compound probe contained one resolvable and one ambiguous field.
  Whole-binding freezing rejected the safe partial correction twice and
  prevented all Actor execution.
- `024`: correctly distinguishing `&&` from background `&` exposed an invalid
  setup contract. A foreground `npm start` blocked until timeout.
- `103: .56629 -> .35`: the generalized audit found incomplete CAD geometry
  but report-only routing suppressed the useful repair used by the baseline.
- `050`: completion rewrote previously passing REAPER structure without backup
  or rollback protection.

### Mechanisms that did not deliver

#### Probe disposition

- Whole-binding locking was too coarse.
- Compound probes could not be partially resolved.
- Unresolved exact claims could remain in the binding sent to Actor.
- Evidence validation established step membership, not semantic support.
- Task `042` demonstrated a complete protocol-caused task loss.

The safety objective remains reasonable; this implementation must not be
reused.

#### Compact manifests

No scored gain was causally attributed to the structured manifest mechanism.

- `046` already had the exact 25+8 file set in the selected baseline. Its gain
  came from an evaluator-recognized XCF layer name.
- `062` improved because the Actor edited the evaluator-facing snap Zotero
  database. The selected baseline had exhaustive bindings but edited a
  different database and received `2/86` attachment matches.

#### Per-proof provenance

- Conservative unknown reports avoided speculative repair on some controls.
- Schema failures could collapse useful findings into synthetic all-unknown
  reports.
- `019,055,060` produced forced reports.
- `055` lost a useful recheck because `fail` plus `blocked_by` violated the
  schema.
- Task `100` proved that a nominally read-only Verifier can still mutate
  evaluator-visible state through its inspection method.

#### Completion routing

- It activated only after the original Actor submitted and a full audit ran.
- `047` and `055` reached completion after roughly 188-195 Actor calls and had
  only about 20 calls left.
- `047,055` remained zero.
- `048,056` were infrastructure-censored.
- `050` damaged passing state.

Completion was immediate after detection but detection occurred too late in the
task lifecycle. More calls alone would not resolve the underlying domain and
preservation failures.

#### Rollback

- `019` proved that a regular-file backup could be restored.
- The repair's remote upload/publication was not reverted.
- Completion mutations were not protected.
- Pass-to-unknown rechecks were treated as safe.
- Exceptions during repair/recheck could bypass restoration.

Rollback was not useless, but it was too narrow to establish end-to-end safety.

#### Native/accessibility-first guidance

`053` and `059` had favorable Actor trajectories, but neither isolated the
prompt as the cause. The guidance remains sensible, but no aggregate effect was
demonstrated.

### Infrastructure findings

The actual boot-QCOW expansion worked, and root-readonly detection was useful.
They did not solve the main reliability classes:

- `021`: OSWorld-web state initialized without a cookie.
- `024`: foreground service lifecycle timeout.
- `048`: repeated guest command-service loss.
- `056,060`: ext4 remounted read-only despite approximately 49G free.
- `082`: nested Docker deterministically broke the guest controller channel.
- Initial `071,090` attempts also encountered read-only roots.

Historical Actors sometimes responded to an already broken root with remount,
SUID, polkit, password, or privilege-escalation-style commands. Those were
unsafe downstream reactions, not demonstrated causes of the original ext4
remount. Future harnesses must abort immediately on `ro`/`EROFS`.

Width 54 added 193 model-response retries and seven terminal task failures. It
must not be used for ordinary scored comparisons.

## Which gains were real

### Strong mechanism-consistent signals

- `062: +.23165`: active canonical profile/database resolution changed Zotero
  attachment matches from `2/86` to `63/86`.
- `080: +.04691`: a targeted probe resolved generic versus segment-specific
  Sales AE role serialization and produced the correct formula-derived total.

These are promising single-task signals, not repeated causal proof.

### Conditional repair signal

- `019`: the repair likely added about `.10` by publishing the completed video.
  Recheck exhausted, local rollback restored the file, and remote mutation
  remained. This demonstrates capability but not safe repair.

### Infrastructure signal

- `081: 0 -> .85`: service startup and evaluator-surface recovery worked
  repeatably. This is infrastructure progress, not improved reasoning.

### Non-attributable gains

- `010`: exact comma-formatting difference.
- `015`: stochastic omission of an extra paragraph.
- `046`: XCF layer-name acceptance.
- `053`: better but still overbroad masking.
- `059`: different Actor strategy producing a minimally accepted GGB.
- `085`: continuous artifact-score variance.
- `086`: infrastructure exposed an incorrect artifact; older profiles already
  reached `.1`.
- `092`: evaluator visual-model execution variance.

## Mistakes not to repeat

1. Do not deploy several unproven schemas and routing policies in one panel.
2. Do not let a Verifier inspect the judged session through a mutating control
   surface such as CDP.
3. Do not freeze an entire binding for a field-level ambiguity.
4. Do not pass an unresolved exact claim to Actor as authoritative text.
5. Do not convert a malformed final Verifier report into all-unknown while
   discarding already validated findings.
6. Do not start completion only after the original Actor has exhausted almost
   all useful execution budget.
7. Do not let completion rewrite previously passing native structure without
   pre/post invariant snapshots and rollback.
8. Do not call local-file restoration end-to-end rollback when remote,
   database, browser, or application state remains mutated.
9. Do not infer mechanism success from an aggregate gain before tracing the
   exact score-changing artifact.
10. Do not mix a high-concurrency stress test with a causal harness comparison.
11. Do not treat evaluator-specific formatting luck or evaluator execution
    variance as behavioral uplift.
12. Do not interpret infrastructure recovery as model or prompt improvement.

## Evidence-supported next levers

Future changes must be independent micro-experiments on top of baseline
behavior `b091c78`.

### Lever 1: canonical active-surface identity

Evidence: `062`.

Before mutation, Actor should prove the active application/profile/database and
the evaluator-facing persistent path using task-visible state. This should be a
small Actor invariant, not a task-specific path table or a new model role.

Test first on:

- `062` as treatment;
- protections with initialized profiles/accounts;
- controls where alternate profiles or copied databases would be harmful.

### Lever 2: field-scoped ambiguity state

Evidence: positive `080`, failure `042`.

Represent a probe against one explicit field or JSON path. A resolved field may
change with direct evidence; unresolved fields remain unchanged and
non-authoritative. Do not require a large parallel resolution schema.

Test offline against recorded Binder/Critic traces before any VM panel.

### Lever 3: observational Verifier boundary

Evidence: `100`.

The Verifier must not use CDP, reload, direct application mutation, or any
control surface that changes evaluator-visible telemetry. Prefer static artifact
copies, read-only parsers, screenshots, accessibility reads, or an isolated VM
clone. Add a regression test proving pre-audit and post-audit evaluator-visible
state are identical.

This is a required safety correction, not an expected score lever.

### Lever 4: preserve ask-user-once prerequisite recovery

Evidence: `095`.

Missing task inputs should route to the baseline's legitimate one-shot recovery
path. Verify exact filenames, hashes when available, and decodability before
resuming. Never turn an unresolved prerequisite into silent submission.

### Lever 5: one bounded missing-action repair

Evidence: `019`.

Only attempt repair when:

- the main artifact already exists;
- one directly observed action is missing;
- all passing invariants can be snapshotted;
- every mutated surface is reversible;
- recheck has a reserved correction call.

Remote or browser-state repair remains report-only until reversible semantics
exist.

### Lever 6: infrastructure preflight and service contracts

Evidence: `021,024,048,056,060,081,082,086`.

- Require explicit background-service declarations and readiness endpoints.
- Preflight OSWorld-web cookies, required HTTP services, root writability,
  evaluator upload, and controller health before Actor starts.
- Abort immediately on `ro`/`EROFS`.
- Isolate nested-Docker tasks.
- Run scored panels around width 23-24.

Keep infrastructure changes outside the behavioral A/B whenever possible.

## Next experimental order

The subsequent full 106-task compact-trace audit is
`docs/CLI_HARNESS_BASELINE_106_COMPACT_TRACE_AUDIT.md`. It ranks isolated
canonical closure before accepted submission as the first behavioral
experiment.

1. Baseline plus canonical closure only; run its treatment, protection, and
   stability panel.
2. Baseline plus observational-Verifier safety only if further Verifier work is
   proposed; task `100` remains the mandatory protection.
3. Offline field-scoped probe replay; only then a small VM panel containing
   `080`, `099`, and `042`.
4. Baseline plus bounded reversible missing-action repair; test `019` and
   protection tasks.
5. Run infrastructure corrections separately from behavioral comparisons.

Promotion requires every full-score protection to remain full score, no
evaluator-visible Verifier mutation, no task lost before Actor due to schema,
and a repeated score gain on more than one independent task.
