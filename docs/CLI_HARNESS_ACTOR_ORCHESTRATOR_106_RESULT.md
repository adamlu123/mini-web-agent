# Actor-Orchestrator 106-Task Result

## Result

The Actor-orchestrator profile achieved:

```text
63.88916151667973 / 106 = 0.6027279388366013
```

Accounting excludes the private GitLab tasks `026,041`, matching the pinned
55% baseline profile.

- Evaluated: `104/106`
- Unevaluated and counted zero: `021,082`
- Strict `1.0`: `24/106`
- Positive scores: `92/106`
- Zero scores including unevaluated tasks: `14/106`

The pinned baseline was:

```text
58.33904089974321 / 106 = 0.5503683103749359
```

The strict improvement is:

```text
+5.550120616936518 total
+0.05235962846166526 per task
+5.23596284616654 percentage points
```

Task-level comparison:

- improved: 34
- regressed: 28
- unchanged: 44

This is the highest completed strict 106-task score currently recorded for the
CLI harness and is the recommended reproduction profile. Its material task
regressions and mixed result provenance remain part of the record.

## Architecture

The persistent main Actor:

1. chooses Binder, Critic, and Verifier invocation budgets;
2. reads their reports as tool outputs;
3. performs all intended task mutations itself;
4. decides when to verify, repair, re-verify, and submit;
5. manages one visible hard 300-call pool including subagents and compaction.

The runner enforces only the hard total, per-invocation caps, role ordering, and
tool permissions. It does not reserve a Verifier budget.

All Responses API calls include:

```json
{
  "strict_session": true,
  "infrastructure": "aiservices"
}
```

AIServices eliminated the HTTP 429 saturation seen in the first width-27
attempt.

## Behavior provenance

Single-phase Actor-orchestrator behavior:

- commit: `e727ca43`
- includes the Actor-orchestrator behavior from `590cf427`
- includes AIServices routing

Multi-phase task `069`:

- commit: `dbaf2a54`
- same persistent Actor and shared 300-call budget across all four phases
- score: `0.8708`, versus baseline `0.525`

The result is therefore mixed-provenance in one narrow respect: task `069`
required the later multi-phase support commit. Do not describe this as a
single-commit full rerun.

## Result roots

Initial 27-task panel:

```text
results-actor-orchestrator27-aiservices-w27-20260818T201204Z
```

Retries for `088,100,102`:

```text
results-actor-orchestrator27-aiservices-retry3-w3-20260818T230645Z
```

Remaining 79 tasks, width 36:

```text
results-actor-orchestrator79-aiservices-w36-20260819T051423Z
```

Low-width retries for remainder failures:

```text
results-actor-orchestrator79-aiservices-retry11-w6-20260819T094218Z
```

Multi-phase task `069`:

```text
results-actor-orchestrator069-multiphase-aiservices-w1-20260819T094549Z
```

Final setup retries for `021,082`:

```text
results-actor-orchestrator-final2-aiservices-w2-20260819T124744Z
```

The exact merged task map is preserved outside the repository as:

```text
/home/v-theolee/.copilot/session-state/b1cce478-9e7e-4216-abe8-48c94dcaf5c4/files/actor-orchestrator106-final.tsv
```

## Unevaluated tasks

### `021`

Repeated website setup failed before agent execution because the target site
returned no usable cookie. Strict score: zero.

### `082`

Repeated guest setup/upload attempts failed with connection resets before agent
execution. Strict score: zero. Its baseline score was also zero.

Neither task is interpreted as a behavioral failure, but both remain zero in
the strict 106-task accounting.

## Attribution

The strongest supported gain mechanism is not the visible step counter alone.
It is the flexible pre-submit loop:

```text
Actor-owned orchestration
-> timely Binder/Critic consultation
-> candidate persistent state
-> read-only Verifier inspection
-> original Actor repair
-> optional re-verification
-> submission
```

Examples from the audited 27-task panel:

- `008`: Verifier-driven repair of dates, persistent state, and attachments;
- `081`: missing citation detected and repaired before submission;
- `036`: adaptive second Binder/Critic pass corrected Zotero scope;
- `069`: Actor allocated one budget across four sequential phases and scored
  `0.8708`.

The shared counter helps the Actor schedule execution and feedback, but it is
not independently isolated as causal. Only a minority of major gains activated
the midpoint reminder.

## Risks and regressions

The architecture can also:

- let the Actor spend almost the entire pool before verification;
- accept weak Verifier checks;
- ignore correct specialist warnings;
- persist with falsified visual strategies;
- produce large task-level variance.

Audited examples include:

- `039`: incomplete late-state coverage after exhausting the budget;
- `100`: no Verifier call and forced submission with an incomplete puzzle;
- `102`: image-only DOCX files accepted despite an editable-content requirement;
- `062`: weak Zotero result despite a `ready` Verifier report.

Across all 106 tasks, 28 regressed relative to the selected baseline attempts.
The aggregate uplift is meaningful, but the profile is not protection-safe.

## Decision

Pin this result as the default reproduction profile under
`cli-harness-v2-60pct-106-repro-v1`. The tag contains the complete
single-/multi-phase behavior and corrected setup instructions.

The next refinement should preserve Actor-owned allocation while enforcing:

- a minimum unspendable Verifier reserve;
- substantive native-semantic verification rather than file validity alone;
- submission gating on a real Verifier inspection;
- bounded stop/replan behavior before budget exhaustion.

The previous fixed-pipeline control remains archived under behavior `b091c78`
and reproduction tag `cli-harness-v2-55pct-106-repro-v7`.
