# Binder native-field gaps: tasks 042, 044, 046, 049

## Overall finding

These tasks separate task-visible native contract gaps from evaluator-only
representation predicates.

### Task 042

- Critic marked the final-cut probe unresolved, but adjudication promoted it to
  an exact binding.
- Task-visible native proof should cover playlist-entry `in/out`, duration, and
  A1 serialized audio role—not merely producer metadata.
- Actor omitted or misplaced these fields.
- Evaluator’s required logo rectangle is an evaluator-only geometric constant.

### Task 044

- Task explicitly allowed a full-width black bar.
- Actor used `avfilter.drawbox`, preserving dimensions and resolution.
- Evaluator awarded crop/no-stretch credit only to `crop/qtcrop`.
- Binder was semantically correct; evaluator encoded one representation.

### Task 046

- The complete Sarah indoor/outdoor filename manifest was task-visible and
  should have been exhaustively bound.
- Critic identified uncertainty but final binding released “all and only”
  without a supported manifest.
- Actor created an incomplete/guessed set.

### Task 049

- Task-visible proof should bind connector endpoints as formulas anchored to
  native box centers and verify saved relationships.
- Actor used arbitrary offsets and missed most anchors.
- Curved bypass connection-ID details were evaluator-specific and not explicit
  in the task.

## Reusable principles

1. Bind the serialized location, not only the semantic concept.
2. Enumerate finite sets; unknown manifests remain unresolved.
3. Separate valid semantic alternatives from evaluator-only representations.
4. Pair proof clauses with parser-visible predicates.
5. Inspect final native state after save/reload.
6. Never convert uncertainty into a plausible exact fact.

## Evidence

- `reliability64-alignment-audit/task-042.md`
- `reliability64-alignment-audit/task-044.md`
- `reliability64-alignment-audit/task-046.md`
- `reliability64-alignment-audit/task-049.md`
- `/home/v-theolee/osworld-harness-handoff.md`

