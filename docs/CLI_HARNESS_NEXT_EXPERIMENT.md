# Next CLI Harness Experiment

> **Historical proposal, completed and rejected.**
>
> This file records the rationale and intended design of
> `cli-harness-v2-generalized-next`. The 60-task result did not pass promotion,
> and exact-trace attribution found direct harness regressions. Do not use this
> package as the next development plan. See
> `docs/CLI_HARNESS_GENERALIZED_60_POSTMORTEM.md` for the superseding decision,
> failed assumptions, and baseline-derived next levers.

## Current pins

- Default profile: `cli-harness-v2-55pct-106` at `c9c4ee4`.
- Offline audit: `cli-harness-v2-55pct-106-audit-v1` at `7504c55`.
- Harness behavior: `b091c78`.
- Observed 106-task result: `58.33904`, effective partial `0.55037`,
  strict `19/106`, estimated GPT-5.5 cost `$13.80/task`.
- Provenance is mixed: initial panels at `ff500d9`, gateway retries at
  `b091c78`; no single-commit 106-task rerun exists.

## Causal standard

Do not call an idea invalidated unless its intended mechanism actually
activated under a reasonable implementation and the result contradicted the
hypothesis without dominant confounds.

### Genuinely invalidated

- Actor self-reported pass/unmet/blocked as authoritative evidence.
- Broad Verifier-owned semantic repair without rollback.
- Generic checklists or a fresh Verifier inferring evaluator-only constants.
- Elapsed-budget submission gates.
- More calls alone as an alignment mechanism.
- Script success/screenshots as sufficient final-state proof.

### Failed implementation, not invalidated idea

- unresolved probes were prompt-only and could disappear at adjudication;
- exact-set checking lacked an enforced manifest;
- `prove` clauses were prose and not required to execute;
- final readback was advisory;
- missing whole tasks were routed into small repair rounds;
- recheck could report regression but could not rollback.

## Strongest established targets

### Binder/Critic

- `102`: all ten documents were handled, but the Verifier predicates were
  weaker than the evaluator's exact heading/anchor predicates. Representative
  coverage was not the accepted-trace cause.
- `046`: the selected baseline already resolved and executed the exact 25+8
  file set. Its score difference came from an evaluator-sensitive XCF layer
  name, not exact-set resolution.
- `042`: the generalized whole-binding probe lock prevented all Actor
  execution. Any future probe representation must be field-scoped.

### Actor

- `049`: wrong native connector relationships/geometry.
- `053`: direct visible mask defects survived submission and repair.
- `059`: no canonical `.ggb`; modality existed, but strategy thrashed.
- `062`: exhaustive Binder requirements existed and the selected evaluator
  found `2/86` valid attachment relations because Actor edited a
  non-evaluator-facing Zotero database. Canonical active-surface identity, not
  manifest coverage, was the useful lever.

### Reject as clean optimization targets

- `035,051,054,089` are substantially confounded by evaluator contradiction,
  proxy metrics, missing helper assets, or hard-coded paths.
- `018` contains real process violations but also a hidden evaluator-time
  conflict; do not treat it as a clean Actor target.

## Why high-score controls work

Successful tasks repeatedly show:

```text
exact task-visible contract
-> direct native mutation
-> canonical parser/API/hash/render readback
-> early submission once persisted state is proven
```

Preserve:

- initialized account/profile/browser session;
- exact paths, sets/cardinality, and forbidden extras;
- native/editable artifacts and ordered handoffs;
- source/baseline preservation;
- save/reload/readback;
- ask-user-once recovery;
- flexible Bash/Python/native apps and xdotool when needed;
- anti-proxy/no-hidden-state rules;
- early submission when proof is complete.

## Historical implementation delta

The following package was implemented and tested. It is retained for
provenance, not recommendation. Its combined deployment made causal attribution
harder and introduced failures documented in the postmortem.

No new model role, budget change, broad checklist, self-ledger, timing gate, or
direct Verifier repair.

1. **Infrastructure**
   - fix setup shim so `&&` is not treated as background `&`;
   - fail setup on nonzero return codes;
   - classify root `ro`/`EROFS` and missing required services as infrastructure;
   - retry evaluator postconfiguration transport failure;
   - preserve controller/QEMU/kernel diagnostics before teardown.

2. **Critic probe disposition**
   - final adjudication submits a compact `probe_resolutions` array;
   - every active probe is `resolved`, `unresolved`, or `conflict`;
   - resolved requires evidence steps; unresolved/conflict cannot disappear and
     never triggers repair.

3. **Compact optional manifest**
   - add `expected_count` and `expected_items`;
   - only for finite task-visible repeated sets;
   - max 128 compact identities, no repeated descriptions;
   - dynamic manifests remain mutable until their visible update window closes.

4. **Verifier proof provenance**
   - add `prove_statuses`, `evidence_steps`, and optional `blocked_by`;
   - pass requires every Binder `prove` clause to pass;
   - missing evidence is unknown; unknown is non-mutating;
   - collapse downstream failures behind one root cause.

5. **Post-audit routing**
   - missing major artifact -> completion mode;
   - existing artifact + localized direct failure -> repair mode;
   - unknown/conflict -> report-only;
   - infrastructure -> clean retry.

6. **Scoped repair protection**
   - optional absolute `repair_paths`;
   - back up existing file/database targets and fingerprint related passing state;
   - target must improve and protected passes must remain pass, otherwise restore;
   - no automatic browser/session repair without a safe reversible operation.

## 60-task panel

### Core treatments

`042,046,049,053,059,062,102`

### Full-score protections

`001,003,006,012,013,028,030,039,052,065,066,068,073,095,100,107`

### Stability controls

`004,005,011,015,038,069,072,085,093`

### Evaluator-conflict controls

`010,021,024,035,037,044,080,092`

### Secondary treatments

`016,019,040,047,048,050,055,056,090,103`

### Infrastructure canaries

`060,071,081,082,086`

### Diversity controls

`002,008,014,029,074`

Run GPT-5.5 xhigh, 300 calls, width 23-24, unique result root, exact commit
manifest, and no GitLab tasks `026,041`.

## Promotion gates

- at least 3/7 core treatments improve;
- core-treatment aggregate improves by at least `.75`;
- all 16 protections remain `1.0`;
- no stability control drops more than `.10`;
- stability-control aggregate does not decline;
- conflict controls receive no evaluator-specific destructive repair;
- unknown-only audits make zero mutations;
- every repair records proof provenance and backup outcome;
- infrastructure failure rate and protection median calls do not increase.

If any protection gate fails, retain `cli-harness-v2-55pct-106` as default.

## Infrastructure notes

- `082`: nested Docker startup repeatedly severs the guest controller channel.
- `086`: ext4 remounted root read-only under width 48; width-one setup remained
  writable. The shim's accidental async `npm install` behavior amplified I/O.
- `060,071,086`: root read-only failures must stop the Actor immediately.
- Scored runs should use width 23-24; width 48 saturated the gateway.
