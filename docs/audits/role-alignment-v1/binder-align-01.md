# Binder/Critic banked gains: tasks 012, 015, 027, 028

## Task 012

Binder/Critic resolved the correct Q2–Q5 source identities, replacing wrong
distractors. Actor copied and byte/hash-verified the exact files. Evaluator
scored hidden MD5 matches; semantic/source-preservation clauses were broader
than the evaluator.

## Task 015

Binder captured editable structure and calculations. A historical Critic patch
corrected raw length readings and propagated the mean/modulus, producing a
near-full result. The selected final run later regressed to an unresolved/wrong
reading, showing exact source interpretation remains critical.

## Task 027

Binder captured workbook scope and formulas. Critic correctly recognized that
sample versus population standard deviation was unresolved. Actor followed the
sample convention; daily workbook passed, summary failed.

## Task 028

Historical Critic correction expanded a three-vaccine plan to the required four
bookings. Selected final Binder accurately represented the persistent booking
state and no-extra constraints; Actor completed it for full score.

## Generalizable principles

- source identity and exact naming;
- native editable structure;
- exact sets and no extras;
- propagate corrected source facts through derived values;
- preserve ambiguity when conventions are not task-defined;
- verify final persistent state.

## Evaluator-specific limitations

- hidden MD5 equality;
- literal gold cells;
- population/sample convention;
- exact receipt filenames/date offsets.

## Evidence

- selected paths in `baseline106-selected-attempts.tsv`
- paired historical paths in `winning64-pairs.tsv`
- `reliability64-alignment-audit/task-012.md`
- `reliability64-alignment-audit/task-028.md`

