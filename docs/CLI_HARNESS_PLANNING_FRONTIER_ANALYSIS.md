# Planning, Thrash, and Persistent Frontier Analysis

## Questions

This analysis answers:

1. Does Binder already perform planning?
2. What would a distinct formal Planner add?
3. What do high-call, low-score Actors spend their calls on?
4. Would doing more work per command or model turn help?
5. Does an execution-strategy checkpoint have score evidence?
6. Is the persistent sub-0.5 frontier primarily a planning frontier?

The method follows the process that originally recovered score:

```text
exact selected traces
-> first causal divergence
-> high-score control comparison
-> one repeated invariant
-> focused protection-first panel
```

## Binder versus Planner

Binder already performs:

- acceptance planning;
- task-visible fact resolution;
- exact path/surface identification;
- preservation and no-extra requirements;
- ordered binding dependencies;
- editable -> export -> downstream handoffs;
- observable proof definitions.

Critic challenges those facts, and Binder adjudication owns the final contract.

Actor already performs:

- tool and application selection;
- native versus UI/direct-artifact strategy;
- concrete execution;
- source recovery;
- save/reopen/readback;
- tactical fallback.

A distinct Planner would only add:

- an action-level route before execution;
- the first persistent artifact milestone;
- tool-family commitment and switching criteria;
- bounded alternatives;
- mid-execution replanning after a failed prototype;
- invariant-preserving dependency gates within one binding.

A Planner that restates requirements, facts, paths, or proof clauses duplicates
Binder and risks freezing a wrong contract.

## Quantitative command evidence

Across the exact selected 106-task baseline:

- correlation of score with Actor steps: approximately `-0.43`;
- correlation of score with run-command count: approximately `-0.41`;
- correlation of score with average command length: approximately `+0.12`;
- correlation of score with multi-command rate: approximately `+0.17`.

These are descriptive and complexity-confounded, but they reject the simple
claim that more calls or larger commands generally improve score.

### High-call low-score versus full-score controls

Selected high-call low-score tasks:

`047,048,054,055,056,059,071,088`

| Metric | High-call low-score | Full-score controls |
|---|---:|---:|
| Mean Actor steps | `234.1` | `44.1` |
| Mean run commands | `184.5` | `41.6` |
| Multi-command rate | `88.3%` | `90.4%` |
| Mean image reads | `42.4` | `1.3` |
| Mean compactions | `5.25` | `0.56` |
| Mean response retries | `7.63` | `0.44` |

The failing tasks were not insufficiently batched. They performed many more
already-batched operations.

High-call tasks also lost feedback:

- 63 truncated Actor outputs across 7/8 tasks;
- only approximately 16.1% of command-output characters remained visible;
- controls retained approximately 34.8%.

Large uncertain batches increased blast radius and hid intermediate failures.

## What batching helps

Batching is effective when:

- the target surface is already known;
- inputs and predicates are deterministic;
- output is bounded;
- assertions and rollback/readback are included.

Examples:

- `001`: one bounded calendar backup/write/readback operation;
- `012`: exact copy plus hash and set assertions;
- `022`: known document/hash uploads followed by state reload;
- `107`: archive extraction, netlist, DRC, and identity checks.

Batching is harmful or neutral when:

- GUI actions require state feedback;
- geometry or masks are guessed;
- source evidence is unavailable;
- native serialization is being reverse engineered;
- a monolithic script combines construction, rendering, and validation.

Examples:

- `047`: a 639,976-character Unity asset dump retained only 15,086 characters;
- `054`: repeated 5–10 KB compositor rewrites amplified bad masks;
- `055`: one script combined mask generation, MLT construction, render, and
  probe, producing truncated output;
- `088`: a 678-line generator created outputs but not functional native
  controls.

Therefore:

> Batch deterministic work after route selection; keep uncertain actions in
> short feedback loops.

## Thrash after a known blocker

Nine selected non-successes spent:

- 1,281 run-command calls after the first clear blocker;
- approximately 142 calls per task after that blocker;
- 35 post-blocker compactions;
- approximately 3.9 post-blocker compactions per task.

Compaction frequently preserved the failed strategy:

- `047`: registry variables did not advance the game, but graph/asset probing
  continued;
- `048`: failed drags were followed by assembly/static inspection rather than
  calibrated interaction;
- `050`: source 403s were followed by long searches and synthetic audio;
- `055/056`: failed visual/MLT approaches persisted through repeated renders;
- `059`: valid temporary GeoGebra state existed, but JAR/XML probing continued
  instead of canonical import/save;
- `071`: root was known read-only, yet mount/polkit recovery continued;
- `088`: a visible macro button failed, but generator experiments continued.

Successful controls used compaction differently:

- `039`: preserved a working moderation/API route and continued state mutation;
- `095`: preserved legitimate source recovery and moved to native project
  verification.

The repeated difference is not having a todo list. It is whether a failed
causal premise is invalidated.

## Persistent sub-0.5 frontier

Across v1, reliability, xhigh, Fresh Verifier, Actor-repair, generalized, and
canonical-closure results:

- 25 tasks stayed below `.5` in every comparable profile with at least three
  observations;
- 27 tasks were below `.5` in at least 80% of comparable profiles;
- selected Actor-repair mean across those 27 tasks: approximately `.1179`.

Persistent frontier:

`018,021,023,024,033,035,036,037,042,044,047,048,049,050,054,055,058,059,060,061,067,080,086,088,089,090,102`

### Non-exclusive causal projection

- strict broad-Planner-eligible: 1 (`018`);
- narrow invariant candidates: 6 (`018,042,049,088,090,102`);
- evaluator/task conflict: 11;
- domain/reference limited: 10;
- infrastructure: 5.

This does not mean planning is useless. It means the persistent frontier is not
primarily one planning failure mode.

## Frontier classes

### Evaluator/task conflict

Examples:

- `021`: hidden Balcony 402 seat set versus task-supported 405;
- `023`: hard-coded Miniconda/ROBOX/temp-file gate;
- `024`: visible truthful `$12,000` versus hidden `$18,000`;
- `035`: `Salesforce License renewal` versus evaluator label;
- `037`: hidden case-sensitive BibTeX keys;
- `044`: valid black bar versus crop-filter-only proxy;
- `061`: hidden pixel reference;
- `080`: task/business semantics versus evaluator memo;
- `089`: missing/hard-coded reference root.

A Planner cannot solve these without evaluator-derived constants.

### Domain capability

Examples:

- `047/048`: game completion;
- `050`: authentic audio acquisition and native edit semantics;
- `054`: visual matting;
- `055`: reference video replication;
- `067`: polyphonic transcription;
- `090`: complex visual/native slide geometry.

Planning may stop waste or improve sequencing but does not provide the missing
domain solution.

### Infrastructure

Examples:

- `060/086`: read-only root/evaluator upload;
- `048/058`: command-service failure in some attempts;
- `024/089`: setup/service/reference failures.

Planning should abort unsafe recovery and classify these separately, not claim
behavioral uplift.

### Narrow native or state invariants

Examples:

- `042`: exact MLT playlist timing/fade fields;
- `049`: connector endpoint/relationship geometry;
- `088`: functional native ActiveX/control serialization;
- `090`: rendered slide anchors plus native geometry;
- `102`: exhaustive per-document heading/anchor/format matrix.

These do not share one implementation. They support focused task-class
experiments.

## What has actual improvement evidence

### Canonical active-surface identity

Task `062` improved approximately `+.23165` when Actor used the
evaluator-facing Snap Zotero database rather than another database.

This is direct mechanism-consistent score evidence.

### Legitimate source recovery

Task `095` reached full score by asking once, receiving the legitimate source,
downloading exact assets, and verifying them before construction. Suppressing
that recovery caused a later `1 -> 0` regression.

This is strong safety and process evidence.

### Early canonical artifact

Task `059` repeatedly moves from `0` to `.2` when a native `.ggb` exists, but
the construction route varies and later semantic gates still fail.

This supports artifact existence as a necessary gate, not a proven Planner
uplift.

### Targeted field ambiguity

Task `080` showed a small deterministic improvement when a specific role-field
ambiguity was resolved. Generic planning is not the causal claim.

## What remains unproven

- a formal Planner as a broad score lever;
- a general execution-strategy checkpoint;
- route replanning as a score mechanism;
- visual anchor prototypes as an aggregate mechanism;
- increased batching;
- formal todo lists;
- more planning calls.

The post-submit canonical-closure experiment is the warning:

- closure activated on 40/42 tasks;
- score declined `-1.58320`;
- no score-changing improvement was caused by closure;
- decisive artifact differences already existed before intervention.

An intervention can be associated with the right principle and still act at the
wrong time or granularity.

## Formal Planner conclusion

Do not implement a general Planner yet.

The strongest argument for planning is not initial planning. It is:

> invalidate a disproven execution premise and choose one bounded alternative.

However, strict frontier classification finds fewer than three independent
tasks with the same broad Planner-treatable first divergence.

A formal Planner would also risk repeating:

- broad checklist/schema expansion;
- fixed planning overhead;
- Binder duplication;
- freezing unresolved or evaluator-conflicted facts;
- reduced Actor budget;
- self-certification without external evidence.

## Evidence-supported next work

Do not run one broad experiment. Run focused eligibility analyses and only then
small panels.

### 1. Canonical profile/database continuity

Treatments:

`036,062`

Protections:

`013,022,052,065,073`

Activation:

- resolve launched app profile/database before first write;
- mutate only that surface;
- restart/reload;
- read back the same surface.

This is the only candidate with direct related score evidence (`062`).

### 2. Browser/session continuity

Treatments:

`020,098`

Protections:

`013,052,073`

Activation:

- same initialized origin/cookie/storage namespace;
- exact persisted leaves/cart after reload;
- no alternate browser/profile/backend.

### 3. Failed-route invalidation

Offline eligibility candidates:

`047,048,059`

Required shared divergence:

- a state-changing route fails twice;
- subsequent trace repeats observation or the same action family;
- no canonical progress occurs.

Only if all three satisfy this exact pattern should a VM panel be run.

Proposed invariant:

> After two deterministic failures of one route, invalidate that route, choose
> one bounded alternative, and require an observable native state transition.

Protections:

`039,095,100`

### 4. Visual anchor prototype

Offline eligibility candidates:

`049,053,054,055,056,090`

Required shared divergence:

- large-scale construction begins before one evaluator-facing anchor is
  proven.

Only then test one anchor before scaling:

- one connector;
- one mask timestamp;
- one clean cutout;
- one transition/title frame;
- one rendered slide.

Protections:

`003,066,085,095,100`

### 5. Source/recovery policy

Treatments:

`011,050,067`

Protection:

`095`

Activation:

- required source demonstrably blocked;
- one bounded legitimate recovery;
- no synthetic substitution;
- stop/ask if recovery fails.

### 6. Exhaustive repeated-output binding

Offline eligibility:

`102` plus at least two other tasks with independent repeated binary outputs.

Do not revive structured manifests until three tasks show the same
task-visible coverage failure.

## Experimental standard

For any candidate:

1. use baseline behavior `b091c78` as null;
2. change one mechanism;
3. run low width;
4. use identical task inputs and final bindings;
5. require external activation evidence;
6. include full-score protections;
7. separate infrastructure;
8. require repeated treatment improvement;
9. reject if the score-changing artifact predates intervention;
10. do not promote from aggregate score alone.

