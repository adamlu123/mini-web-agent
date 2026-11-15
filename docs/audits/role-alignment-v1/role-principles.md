# Cross-role causal principles

## Causal synthesis

The corpus does not support attributing profile-level score changes to the role
architecture alone. The 28-way audit classified representative failures as:

- evaluator/config/proxy-dominant: 14
- Actor-dominant: 13
- mixed Binder+Actor: 5
- domain-limited: 3
- infrastructure/model-refusal: 5

## Established principles

- Pre-Actor Critic review helps with bounded, exact discoveries and is safer
  than post-Actor semantic repair.
- Unresolved facts must remain unresolved; plausible exact adjudication can
  cause Actor and Verifier to consistently execute/check a wrong value.
- Actor self-ledgers are not trustworthy score authorities, but removing
  execution pressure entirely also permits missing artifacts.
- Canonical visible session/profile, native parser-visible state, and
  reload/readback matter.
- Fresh Verifier semantic repair is not established as safe.
- More steps, arbitrary timing gates, and larger Binder budgets are not
  alignment solutions by themselves.

## Remaining Binder/Critic–Actor misalignments

- Wrong binding followed faithfully: task 034 retained a typo and weakened
  exact date/cap facts.
- Evaluator-sensitive omission: task 023 omitted `torchaudio`, while hidden
  temp-file and VM-response gates independently forced zero.
- Correct binding, Actor did not execute: task 011 created neither workbook nor
  Zotero records.
- Assertion substituted for evidence: task 029 copied a marker vector without
  live QA.
- Critic over-specification: task-faithful workflow/preservation clauses can be
  evaluator-invisible.

## Evaluator-blindly solvable gaps

- wrong path or filename;
- missing deliverable class;
- replacement browser/session/profile;
- flattened/non-native output instead of editable state;
- failure to save, reload, and read back;
- premature submission;
- ignoring higher-authority task-provided corrections.

## Hidden conflicts not safely solvable

- unrelated hidden temp-file gates;
- hidden ordering requirements;
- evaluator hard-coded to a different profile/database;
- exact gold cells absent from permitted evidence;
- arbitrary evaluator-only constants or keys.

## Domain and infrastructure limits

Visual quality, source ambiguity, and missing evaluator breakdowns limit causal
attribution. Infrastructure failures independently prevent trustworthy Actor
or evaluator evidence.

## Bottom line

Preserve uncertainty, canonical/native final-state proof, bounded execution
pressure, and the prohibition on Verifier-owned semantic repair. The evidence
does not justify a generic checklist, more steps, timing gate, or new
role-based repair mechanism.

## Evidence

- `/home/v-theolee/osworld-harness-handoff.md`
- `docs/CLI_HARNESS_28WAY_AUDIT.md`
- `reliability64-alignment-audit/task-011.md`
- `reliability64-alignment-audit/task-023.md`
- `reliability64-alignment-audit/task-029.md`
- `reliability64-alignment-audit/task-034.md`
- `reliability64-alignment-audit/task-054.md`

