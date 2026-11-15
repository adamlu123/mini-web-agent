# Generalized CLI Harness: 60-Task Width-54 Result

## Result

- Branch: `cli-harness-v2-generalized-next`
- Commit: `0c72d74b3bb40a18ada2e24209a335afe377dfcd`
- Result root: `results-generalized60-w54-fixed-20260813T133828Z`
- Model: GPT-5.5 xhigh
- Budget: 300 calls per VM attempt
- Panel: 60 tasks; concurrency 54
- Evaluated: 53
- Failed-as-zero: `021,024,042,048,056,060,082`
- Score sum: `34.3663280250627`
- Effective mean: `0.572772133751045`
- Evaluated-task mean: `0.6484212834917491`
- Strict full completion: `17/60`

The previous selected-attempt panel baseline was approximately `0.57184`; the
experimental aggregate gain was only about `0.00093`. This is not a promotion.

The exact-trace causal postmortem is
`docs/CLI_HARNESS_GENERALIZED_60_POSTMORTEM.md`. It supersedes optimistic
mechanism interpretations in the original experiment plan. Preserve this
profile only as an archived rejected result; do not use it as the base for
future harness work.

## Promotion decision

Retain `cli-harness-v2-55pct-106` as the default.

Disqualifying outcomes:

- full-score protections `095` and `100` fell from `1.0` to `0`;
- task `042` still failed Critic-probe adjudication;
- diversity and secondary-treatment aggregates declined;
- seven tasks failed before evaluation;
- three Verifier reports were still forced unknown.

Positive signals:

- `062 +0.23165`: canonical active Zotero database alignment;
- `080 +0.04691`: targeted role-serialization probe resolution;
- infrastructure task `081` improved from `0` to `0.85`;
- localized repair on `019` performed the missing publication action, although
  recheck and rollback coverage were incomplete.

Other apparent gains were not clean mechanism evidence: `046` changed an
evaluator-sensitive XCF layer name, `053/059` were favorable Actor
trajectories, and `092` reflected evaluator execution variance.

## Mechanism activation

- Completion continuations: `047,050,055,056`
- Local repair: `019`
- Rollback: one successful restoration (`019`)
- Forced Verifier reports: `019,055,060`
- Recorded infrastructure failures: 3
- Model response retries: 193

Completion routing was neither effective nor sufficiently safe: `047` and
`055` remained zero, `056` failed after the guest root became read-only, and
`050` lost previously passing REAPER structure without rollback protection.

The Verifier was directly harmful on `100`: CDP inspection reloaded the judged
session, set the persistent anti-DevTools flag, and converted a solved state
into an evaluator hard failure.

## Infrastructure failures

- `021`: repeated missing-cookie setup failure.
- `024`: task service setup lost the guest command channel.
- `048`: three consecutive guest command-service failures.
- `056,060`: ext4 root remounted read-only.
- `082`: known nested-Docker controller failure.

The actual boot disk expanded successfully to approximately 79G, but read-only
filesystem failures still occurred, so insufficient root capacity was not the
sole cause.

## Invalid predecessor roots

Do not combine these with the result:

- `results-generalized60-20260813T103052Z`: all attempts failed before model use
  because a recording-cleanup command matched its own shell.
- `results-generalized60-rerun-20260813T103816Z`: stopped at the checkpoint;
  local websites were offline and blanket setup return-code strictness was
  incompatible with legacy task setup.
- `results-generalized60-w54-20260813T111556Z`: completed but invalidated because
  no-probe adjudication and Verifier provenance validation caused harness-level
  failures; corrected in `0c72d74`.
