# Binder hidden conflicts: tasks 021, 023, 024, 035, 037

The audit treats Binder as wrong only when it contradicts task-visible evidence,
not merely when it misses hidden evaluator behavior.

## Task 021

- Visible TeamChat evidence supported Balcony 405, Section 301, and Floor 102.
- Evaluator hard-coded Balcony 402.
- Binder was not substantively wrong, but Critic over-concretized a qualitative
  requirement without visible seat-map evidence.
- Actor separately bypassed normal selection through direct state injection.

## Task 023

- Visible contract supported a Conda ROBOX environment, editable install,
  dependencies, and smoke tests.
- Evaluator additionally required `torchaudio`, `/tmp/tmp_file`, and a
  successful hard-coded command.
- Omitting `torchaudio` was a real coverage gap; the temp-file gate was
  evaluator-only and independently capable of forcing zero.

## Task 024

- Visible evidence required truthful `$12,000` funding, a `$6,000` shortfall,
  completed draft sections, and no final submission.
- Evaluator required a hidden `$18,000` certificate and Downloads export paths.
- Binder was task-faithful; inferring the hidden amount/path would violate
  provenance.

## Task 035

- Final scope omitted two evaluator ground-truth rows.
- Evaluator’s protected-row rule also conflicted with the visible instruction
  to reject the monitor.
- This is mixed: incomplete visible request inventory plus contradictory
  evaluator baseline semantics.

## Task 037

- Actor produced valid papers/citations and a compiled PDF.
- Evaluator required arbitrary case-sensitive BibTeX keys.
- Binder should preserve visible existing keys rather than invent aliases, but
  hidden exact casing cannot otherwise be inferred.

## General evaluator-blind rule

Binder should:

1. bind explicit user requirements and visible state;
2. preserve truthful values, provenance, and existing identifiers;
3. derive concrete values only from determinate visible evidence;
4. prove final state and required process when task-visible;
5. keep grounded, derived, unknown, and conflict facts distinct.

Binder must not guess hidden seat IDs, hidden files, magic paths, fetched gold
workbooks, exact key lists, or evaluator export conventions.

## Evidence

- `reliability64-alignment-audit/task-021.md`
- `reliability64-alignment-audit/task-023.md`
- `reliability64-alignment-audit/task-024.md`
- `reliability64-alignment-audit/task-035.md`
- `reliability64-alignment-audit/task-037.md`

