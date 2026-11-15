# Binder/Critic banked gains: tasks 002, 006, 017, 019

Paired V1 → reliability movement:

- 002: `.50 -> .75`
- 006: `.9444 -> 1.0`
- 017: `.2857 -> .5714`
- 019: `.22 -> .46`

## Task 002

Binder/Critic correctly captured required courses, credit intent, Common Core
selection, and semester split. Actor followed the selected course set.

Remaining failure:

- final backend ICS serialized Sem2 in 2025;
- evaluator uses fixed 2025-Sem1/2026-Sem2 buckets;
- Actor noticed the issue but did not rewrite/reimport.

This is successful semantic binding with incomplete final serialization/readback.

## Task 006

Critic expanded a five-file draft to the exact nine-file final set. Actor
obtained/renamed the sources, removed extras, and verified readability.

The evaluator only checked directory filenames, not deeper eligibility or
provenance. The exact-set correction directly banked the final points.

## Task 017

Binder captured the six stop identities/addresses and Critic added
OIA-before-ID. Actor followed that route logic.

Remaining gaps:

- evaluator expected a different ID/CS order;
- Actor did not fully verify the final active DOM/order and walking mode.

This combines task-faithful correction with evaluator-order conflict and
surface-proof weakness.

## Task 019

Critic fixed the swapped episode 2/3 source mapping and merge order. Actor
followed it, producing correct trims/watermarks/merge/upload.

Remaining loss came from plausible invented subtitle descriptions rather than
accurate dialogue-aligned text. The hidden evaluator used ten semantic
checkpoints.

## Reusable principles

1. Convert evidence into exact sets, cardinalities, order, filenames, and
   negative constraints.
2. Include save/export/import serialization in correctness.
3. Verify source mappings before producing derivatives.
4. Do not replace missing evidence with plausible content.
5. Critic challenges semantic facts and final representation.
6. Actor must read back the final machine-visible surface.

## Evidence

- selected paired directories in `winning64-pairs.tsv`
- `evaluation_examples/task_class/task_002.py`
- `evaluation_examples/task_class/task_006.py`
- `evaluation_examples/task_class/task_017.py`
- `evaluation_examples/task_class/task_019.py`

