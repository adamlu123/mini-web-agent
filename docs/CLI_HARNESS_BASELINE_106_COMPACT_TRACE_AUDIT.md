# Baseline 106-Task Compact-Trace Audit

## Purpose

This audit reapplies the analysis method that originally discovered the
reliability/artifact-first jump:

```text
compact exact-trace audit
-> first-causal-failure classification
-> comparison with high-score controls
-> one concrete invariant
-> matched protection-first experiment
```

It does not infer a lever from aggregate score or from generally plausible
agent architecture.

## Exact accounting

The selected-attempt map was reconstructed from:

- `results-actor-repair48-20260811-134300`
- `results-actor-repair-rest58-20260811-155139`
- `results-actor-repair-gateway12-retry-20260812-005203`

Selection rule:

1. the 12-task gateway root overrides those tasks' original attempts;
2. within the selected root, use the final evaluable retry;
3. if no evaluation exists, use the final failed attempt and count it zero.

The reconstruction exactly reproduces:

- tasks: `106`
- evaluated: `103`
- failed-as-zero: `060,082,086`
- score sum: `58.33904089974321`
- effective mean: `0.5503683103749359`

Twenty-eight GPT-5.6 Luna agents audited disjoint balanced groups covering all
106 selected tasks. Incorrect-provenance reports were rejected and rerun
against the selected baseline directories.

For each task the audit read, as available:

- `bindings-draft.json`
- `critic.json`
- `bindings.json`
- `trace-slim.jsonl`
- `verification-audit.json`
- repair/recheck artifacts
- `evaluation/result.json`
- the task evaluator and directly invoked helpers.

## What the full baseline confirms

The repeated successful pattern remains:

```text
exact task-visible contract
-> correct initialized app/profile/path
-> direct native persistent mutation
-> save/reload/reopen
-> parser/API/hash/render readback of the judged surface
-> submission after persistence is proven
```

Representative controls:

- `001`: exact ICS mutation and event-set readback.
- `003`: native PPTX backgrounds and exact image/filter state.
- `006`: exact nine-file PDF set.
- `012`: exact file hashes.
- `013`: same-session API state and exact numeric allocation.
- `017`: exact active route state.
- `022`: exhaustive 23-document hash/category state.
- `028`: exact persisted booking records.
- `030`: logs, checkpoint, target copy, and DOCX table.
- `039`: exhaustive moderation target set and preserved benign state.
- `052`: initialized TravelHub session and exact checkout state.
- `065`: exact two-leg/two-passenger booking state.
- `066`: native in-place PPTX.
- `070`: complete persisted AWS/TeamChat state.
- `073`: exact ReviewSphere edge set and capacity constraints.
- `095`: legitimate source recovery, native MLT, rendered MP4, and readback.
- `100`: visible same-session puzzle completion.
- `105`: high-quality canonical NIfTI.
- `107`: exact native KiCad project/archive structure.

These controls do not support a new generic Verifier or a broader schema. They
support tighter closure around the actual judged persistent surface.

## Recurring failure clusters

The clusters below are non-exclusive. A task may contain both a generic process
failure and an evaluator/task conflict.

### 1. Canonical surface, handoff, or finalization failure

Strong examples:

- `002`: correct calendar content, but folded `DTSTART` serialization failed the
  evaluator parser.
- `008`: report fields passed, but evaluator-retrievable attachment media did
  not reach the scored surface.
- `020`: visible checkout intent coexisted with malformed persisted cart state.
- `024`: browser state existed, but evaluator-facing Downloads export files did
  not.
- `033`: verifier later saw the artifact/log, while evaluator-time canonical
  visibility was empty.
- `036`: Actor and evaluator read different Zotero databases.
- `057`: project existed but final hosted PDF compilation failed.
- `059`: harness submitted although the required `.ggb` never existed.
- `062`: Actor repaired the active Zotero profile while evaluator read the stale
  snap path.
- `081`: correct local project database scored zero because evaluator-facing API
  service was unavailable.
- `098`: submitted browser state omitted persisted leaves that the Actor had
  entered.
- `108`: repair changed the saved PCB but left the evaluator-facing archive
  stale.

This is the broadest repeated generic cluster with both high-score controls and
clear failures.

### 2. Exact native serialization or geometry mismatch

Representative examples:

- `005`: one exact claim-summary string.
- `027`: sample versus population standard deviation.
- `045`: seven frames versus the evaluator's ten-frame bonus threshold.
- `046`: XCF photo layer name lacked an image extension.
- `049`: connector endpoints missed native box anchors.
- `051`: XCF layer bounds differed materially.
- `055`: native MLT lacked evaluator-required playlist/service markers.
- `058`: slide shape graphs differed and transitions were Fade rather than
  literal Morph.
- `063`: crop representation failed the native predicate.
- `071`: native lyric arrays did not match measure-level syllables.
- `076`: native box spacing missed an exact EMU threshold.
- `083`: filenames did not satisfy the required year/surname/title form.
- `087`: native animation path lengths/counts missed quantitative gates.
- `088`: workbook lacked functional ActiveX/control serialization.
- `090`: visual/native slide geometry diverged.
- `094`: valid CAD semantics missed analytic geometry layers.
- `097`: one year and exact citation anchors failed all-or-nothing buckets.
- `102`: headings, anchors, inline formatting, tables, and lists differed.
- `103`: valid solid used wrong spacings and surface type.

The generic lesson is not to encode evaluator-only constants. It is to read
back every exact task-visible native predicate rather than accepting semantic
or visual equivalence.

### 3. Binder/Critic fact or ambiguity propagated into execution

Examples:

- `007`: final course-count vector was wrong and Actor followed it.
- `015`: unresolved measurements/graph choice became exact document content.
- `018`: selected calendar time conflicted with the evaluator's exact time.
- `031`: final contract explicitly denied two evaluator-required defects.
- `034`: unresolved high-weight date/email/budget fields became exact state.
- `043`: contract omitted preservation of evaluator-designated original events.
- `096`: unresolved chart windows, averages, and baselines remained score
  limiting.
- `099`: Critic correctly identified an unresolved exact coordinate; final
  contract weakened it to semantic similarity and Actor guessed.
- `101`: Binder required lower answer `2`; evaluator required `3`.

The generalized whole-binding probe schema is not the answer. The supported
idea is smaller: one ambiguous field remains explicitly non-authoritative until
Actor resolves that field from task-visible evidence.

### 4. Domain/visual execution capability

Examples:

- `047,048`: prolonged game exploration never reached canonical completion.
- `050`: blocked source acquisition led to synthetic audio substitution.
- `053,054`: visual masking/compositing remained visibly incorrect.
- `055,056`: native video structures existed but did not match required timing,
  geometry, or reference output.
- `067,071`: valid native music files lacked correct audio-derived semantics.
- `075,077,079,090`: content existed but reference-layout/visual structure was
  weak.
- `106`: structurally valid labelmap had poor anatomical segmentation.

Repair did not transform these tasks. More calls alone is not a supported
solution.

### 5. Evaluator/task conflict or hidden predicate

Examples include:

- `009`: contributor classification and PDF hash conflict.
- `010`: paragraph-only exact DOCX comparison despite table-semantic success.
- `021`: hard-coded seat constant.
- `023`: opaque path/helper/temp-file contract.
- `024`: hidden `$18,000` source versus visible `$12,000`.
- `032`: `The Paintress` versus evaluator `Alicia`.
- `035`: `Salesforce License renewal` versus `Salesforce License`.
- `037`: hidden case-sensitive BibTeX keys.
- `040`: preserve Q3 and append Q4 versus evaluator Q4-only totals.
- `044`: semantically acceptable black bar versus crop-filter-only scoring.
- `061`: hidden pixel-reference MSE.
- `072`: regex/file-presence scoring weaker or different from task semantics.
- `078,092`: opaque visual-model branches.
- `084,085`: hidden reference-audio/report similarity.

These are report-only research findings. They must not be converted into
evaluator-derived runtime constants.

### 6. Infrastructure-censored outcomes

- `060`: ext4 root remounted read-only; subsequent privilege-style recovery was
  an unsafe downstream reaction.
- `082`: nested Docker deterministically severed the controller before Actor.
- `086`: root remount/evaluator upload failure censored the selected attempt.

Additional attempts on `048,071,084` also encountered command-service or
read-only failures. Infrastructure must be measured separately from behavior.

## What the audit does not support

The full baseline does not support:

- another broad checklist;
- an Actor self-ledger as authority;
- a fresh Verifier acting as evaluator emulator;
- broad semantic repair;
- more calls alone;
- late completion after the original Actor is exhausted;
- whole-binding ambiguity locks;
- file-only rollback described as end-to-end safety;
- evaluator-specific native readers or hidden constants in the runtime.

## Strongest next behavioral lever

### Canonical closure before accepted submission

The most repeated generic difference between controls and failures is whether
the exact judged persistent surface is complete and directly reread after the
last mutation.

This is narrower than a checklist and does not ask Actor to self-score.

Proposed isolated behavior:

1. Preserve the baseline Binder/Critic/Actor architecture and budgets.
2. On the Actor's first `submit_task`, do not evaluate immediately.
3. Inject one concise final-closure instruction:
   - identify the exact requested persistent target;
   - save it;
   - reload/reopen the same app/profile/path/session;
   - read back the task-visible native fields, set/cardinality, or rendered
     output that establish completion;
   - remove stale alternatives;
   - submit again.
4. Do not require a JSON ledger or pass/fail claims.
5. Do not infer hidden evaluator predicates.
6. Allow immediate second submission when the trace already contains adequate
   canonical readback.

This directly targets:

- missing canonical deliverables (`057,059`);
- stale handoffs/archives (`024,108`);
- malformed or omitted persisted state (`002,020,098`);
- evaluator-unreachable service/attachment state detectable through task-visible
  readiness (`008,081`);
- exact native finalization (`046,049,055,063,083,087,088,097,102,103`).

It will not solve domain capability, hidden evaluator conflicts, or
infrastructure.

### Why this lever ranks above new probe schemas

- It repeats across more independent domains.
- It is already the defining behavior of full-score controls.
- It changes no model role and adds no large schema.
- It can be protection-tested cheaply.
- It preserves ask-user-once source recovery.
- It does not authorize Verifier mutation.

## Secondary candidates

### Field-scoped ambiguity state

Supported by `099` and partial evidence from `034,096`; safety constraints are
informed by `042`. Test offline first. Do not deploy the generalized
whole-binding protocol.

### Source-provenance recovery

Preserve the baseline ask-user-once path demonstrated by `095`. Add no new
mechanism until source-blocked failures such as `050` are tested against the
same legitimate recovery opportunity.

### Infrastructure preflight

Separately test service/cookie/root/evaluator-upload readiness and nested-Docker
isolation. Do not mix these changes into the behavioral A/B.

## Required first experiment

Run baseline behavior plus canonical closure only.

Suggested panel:

- direct treatments:
  `002,008,020,024,046,049,057,059,083,098,102,103,108`
- full-score protections:
  `001,003,006,012,013,017,022,028,030,039,052,065,066,070,073,095,100,107`
- stability controls:
  `004,005,029,038,045,074,076,085,091,093,105`
- infrastructure excluded from the behavioral A/B:
  `060,082,086`

Run GPT-5.5 xhigh, 300 calls, width 23-24, unique roots, and matched evaluator
inputs.

Promotion gates:

- every full-score protection remains full;
- no protection incurs more than trivial call overhead;
- at least four direct treatments demonstrate the closure activation;
- at least three independent treatment tasks improve;
- aggregate direct-treatment score improves by at least `0.75`;
- no task is lost before Actor/evaluation due to protocol;
- no evaluator-visible state is mutated by Verifier;
- infrastructure failures remain separately counted.

If these gates fail, discard canonical closure as an implementation rather than
expanding it into another package.
