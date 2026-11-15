# Historical Harness Gain Analysis

## Scope

Thirty GPT-5.6 Luna agents manually audited the two historical score jumps:

1. Same 64 tasks:
   - V1: `25.988658 / 64 = 0.406073`
   - reliability/artifact-first package:
     `34.882563 / 64 = 0.545040`
   - delta: `+8.893905`

2. Matched 105 tasks:
   - old xhigh selected profile:
     `50.719463 / 105 = 0.483043`
   - final selected package:
     `57.339041 / 105 = 0.546086`
   - delta: `+6.619577`

The exact selected attempt paths are preserved in the session artifacts:

- `winning64-pairs.tsv`
- `winning105-pairs.tsv`

Three reports that substituted Fresh Verifier for the final Actor-repair
profile were rejected and rerun against the exact selected directories.

The audits were intentionally freeform. They read paired bindings, Critic
artifacts, Actor traces and commands, evaluator outputs/code, infrastructure
logs, and score-facing files.

## Main conclusion

The gains are real package gains, but there is no single dominant lever.

The repeated contributors are:

1. **Binder/Critic exact contract correction**
2. **Actor construction of evaluator-facing persistent state**
3. **Enough execution capacity and pressure to finish real artifacts**
4. **Legitimate source recovery**
5. **Infrastructure and transient-failure recovery**

Original-Actor repair is a secondary, mixed contributor.

## First jump: `0.406 -> 0.545`

### Strongest repeated cause: Binder/Critic exactness

A conservative set of clear task-visible fact/contract corrections accounts
for approximately `+6.118` gross task points:

- `012 +.80`: correct Q4/Q5 source identities
- `015 +.95`: corrected measurements and derived calculations
- `019 +.24`: corrected episode/source mapping and merge order
- `027 +.50`: corrected serialized date form
- `028 +1.0`: corrected four-vaccine inventory
- `032 +1.0`: corrected hard-gate theme/name mapping
- `038 +.903`: expanded from a five-lead subset to complete lead/conversion
  state
- `073 +.133`: removed two reviewer conflicts
- `002 +.25`: corrected the scored Common Core course
- `006 +.056`: corrected the exact nine-file set
- `017 +.286`: improved the ordered route facts

A broader mixed set including `070,034,029` reaches approximately `+6.692`
gross points, but those tasks also contain Actor-execution contributions.

This is gross movement, not an isolated causal estimate; the package also
introduced factual/serialization losses:

- `010: 1 -> 0` from one comma in an exact paragraph
- `046: .50 -> .46` from an incomplete exact file set
- other tasks retained unresolved or evaluator-conflicting facts

Nevertheless, exact Binder/Critic correction is the most repeated clean
mechanism in the first jump.

### Second contributor: evaluator-facing Actor execution

Independent or mixed Actor/artifact improvements account for substantial gross
movement:

- `065 0 -> 1`: correct persisted booking/contact state and final readback
- `068 .16 -> 1`: successful visible game score and screenshot readback
- `064 0 -> .55`: evaluator-compatible native solver/default paths
- `050 .22 -> .67`: more scoreable native REAPER structure
- `078 .35 -> .70`: materially richer visual site
- `053 +.252`: better masking artifact
- `007 +.20`: scored email-body facts added
- `009 +.193`: richer persisted RIMS state
- `061 +.066`: better image transform
- `074 +.041`: exact missing content strings
- `075 +.04`: additional scored site content
- `083 +.05`: evaluator-visible BibTeX tags

The common successful sequence is:

```text
correct contract
-> correct initialized surface
-> real persistent mutation
-> parser/API/hash/render readback
```

However, no single prompt or reminder caused all of these gains. They also
changed reasoning effort (`medium -> high`), total budget (`80 -> 160`),
Binder/Critic lifecycle, and stochastic Actor trajectories.

### Artifact-first pressure: supported as a package property, not isolated

Support:

- V1 frequently submitted missing or shallow artifacts.
- `007,064` continued from initial construction to evaluator-visible state.
- `030` improved deterministic data/checkpoint/readback.
- the low-score audit found 15/27 Actors materially noncompliant.

Counterevidence:

- `011` regressed `.556 -> 0` after spending the expanded run on blocked source
  research and creating no deliverables.
- `048` still never created a completed save.
- `055/063` produced artifacts that remained evaluator-invalid.
- full-score controls such as `001,003,013,022,052` were already artifact-first
  under V1.

Thus artifact-first execution discipline is part of the winning package, but
the audit does not isolate the reminder or stage machinery as the score cause.

### Budget and reasoning

The first comparison doubled calls and raised reasoning from medium to high.
Additional capacity clearly enabled more source inspection, native construction,
and validation on difficult tasks.

It is not a standalone lever:

- high-score tasks sometimes regressed;
- more calls could prolong a wrong strategy;
- exact hidden/evaluator-conflicting facts remained unavailable;
- command-count analysis later found score negatively correlated with Actor
  steps across the selected baseline.

Treat capacity as an enabling resource, not an alignment mechanism.

## Later jump: `0.483 -> 0.546`

### Failure recovery: strong engineering contribution

Decomposition:

- `+4.1189` on 95 tasks evaluated in both profiles
- approximately `+2.5006` net from changed failure recovery

Major recovered task mass included:

- `063 +.60`
- `104 +.761`
- `105 +.990`
- `106 +.35`

This is real system improvement, but not pure agent behavior. It includes disk,
controller, gateway, timeout, retry, and run-survival differences.

### Repair was not the main cause

Among 105 matched tasks:

- 25 tasks with Actor repair: `+.9120`
- 80 tasks without Actor repair: `+5.7076`

Repair-positive evidence:

- `103`: broader CAD regeneration improved global geometry and removed score
  caps
- `104`: repaired a missing/incorrect lower bore, though old was an
  unevaluated failure
- `098`: likely restored additional persistent form leaves
- `108`: improved some routed-net connectivity

Repair-negative or ineffective evidence:

- `009`: author repair worsened evaluator classification
- `053`: mask repair retained the same defects
- `059`: repair failed to produce the required artifact
- `062`: repaired a native database that was not necessarily the evaluator
  database
- `088`: repair targeted a macro defect but did not change score
- `050`: later completion experiments damaged valid structure

Repair remains a weak, conditional mechanism requiring direct pre/post
evaluation and stronger global-state protection.

### Upstream no-repair gains

Major no-repair gains:

- `095 0 -> 1`: legitimate missing-source recovery and complete MLT/MP4
- `017 .143 -> 1`: correct route order
- `016 0 -> .805`: completed MailHub/CareerLink state after the ledger-era run
  ended with all bindings unmet
- `039 .25 -> 1`: complete dynamic comment inventory
- `025 0 -> .724`: parser-visible filled PDF forms
- `040 0 -> .705`: evaluator-recognized formulas/native workbook
- `063 0 -> .60`: recovered/evaluable PPTX geometry
- `091 .305 -> .857`: much stronger native PPTX/chart/layout construction
- `101 .25 -> .75`: two corrected answer fields
- `105 0 -> .990`: infrastructure recovery plus strong NIfTI output

These do not share one immediate mechanism:

- some are Binder inventory/fact corrections;
- some are better Actor construction;
- some are source recovery;
- some are phase completion;
- some are infrastructure recovery;
- some are ordinary favorable trajectories.

### Ledger removal and execution reminders

Task `016` is the strongest signal:

- old ledger-era run ended after 160 Actor calls with all ten bindings unmet;
- final run created the canonical state and scored `.805`;
- repair did not run.

Task `095` also shows the value of retaining ask-user-once source recovery rather
than treating missing inputs as terminal.

The evidence supports rejecting the Actor ledger as an authority and preserving
execution/recovery pressure. It does not isolate a particular reminder string
or prove that additional planning calls caused the gains.

## What did not explain the gains

- **Fresh audit:** usually ran after the scored artifact already existed.
- **Verifier emulation:** repeatedly passed evaluator-wrong state.
- **Generic checklists:** regressed the matched panel.
- **More calls alone:** enabled both completion and longer thrash.
- **Post-submit closure:** later produced no score-changing correction.
- **Structured manifests/provenance schemas:** no causal gain identified.
- **Formal planning/todo lists:** not isolated in the winning comparisons.

## Best-supported lever ledger

### Strong

1. **Binder/Critic exact task-visible contract formation**
   - repeated large, deterministic gains;
   - especially exact sets, facts, ordering, identities, and hard gates.

2. **Correct evaluator-facing persistent surface**
   - same initialized session/profile/path;
   - real native artifact/state;
   - parser/API/hash/render readback.

3. **Infrastructure/transient recovery**
   - substantial measured contribution to final package score;
   - must remain separately accounted from model behavior.

### Moderate/package-level

4. **Execution pressure without Actor self-certification**
   - strong task-level signal on `016`;
   - consistent with the historical low-score Actor-noncompliance audit;
   - exact prompt/stage cause remains confounded.

5. **Legitimate missing-input recovery**
   - strong safety/success evidence on `095`;
   - insufficient repeated treatment evidence for a broad new mechanism.

6. **Sufficient execution capacity**
   - enables hard tasks;
   - not an alignment mechanism and not independently promotable.

### Weak/conditional

7. **Original-Actor local repair**
   - useful on selected bounded native defects;
   - mixed aggregate and real collateral risk.

### Rejected as score levers

- Actor self-ledger
- generic checklist
- broad Verifier semantic repair
- Fresh Verifier as evaluator emulator
- more calls alone
- late completion
- whole-binding probe locking
- proof-schema expansion
- post-submit canonical closure
- increased command batching

## Decision implications

This analysis explains the historical gains better, but it does not reveal an
unimplemented broad lever with demonstrated recoverable points.

The strongest mechanisms are already present in baseline behavior:

- Binder/Critic
- canonical persistent-state guidance
- execution/recovery pressure
- infrastructure retry

The remaining work should not be another bundled implementation. Any candidate
must pass `docs/CLI_HARNESS_RESEARCH_GATES.md`.

Current candidates remain unqualified:

- improving Binder/Critic further requires three repeated residual
  task-visible fact failures with offline counterfactual corrections;
- execution strategy/replanning has trace evidence but no isolated score
  evidence;
- canonical profile continuity has one strong score signal (`062`);
- source recovery has one strong protection/success case (`095`);
- repair remains weak and unsafe.

The acceptable conclusion may be that no remaining broad evaluator-blind
harness lever has enough demonstrated points to justify another panel.

