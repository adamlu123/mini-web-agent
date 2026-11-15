# Actor visual/media strategy: tasks 053, 054, 055, 056

## Task 053

Actor preserved the source/export relationship and metadata, but retained a
coarse rectangle strategy after exposed legs and oversized masks were visible.
Repair preserved structure but did not fix the visual defect.

## Task 054

Actor constructed the composite externally through Python/PIL and used GIMP
mostly as serialization. Iterations improved local regions while damaging
others. Native/editable provenance and clean segmentation remained weak.

## Task 055

Actor copied/remuxed the reference to guarantee visual similarity, then explored
native MLT construction. This produced an invalid provenance/handoff and no
native structural score.

## Task 056

Actor produced the strongest native project, with editable text and animation,
but visible affine geometry defects remained. Repair diagnosed rather than
completing a corrected fresh render.

## Reusable behavior

1. Structural metadata is only a gate, not visual proof.
2. Maintain one canonical editable source and render from it.
3. Validate the actual semantic defect: locality, alpha edges, or frame
   geometry.
4. Repair the causal representation rather than broadening approximations.
5. Visual similarity and native completeness are distinct.

## Evidence

- selected task traces and verification rechecks for 053–056
- `reliability64-alignment-audit/task-054.md`
- `reliability64-alignment-audit/task-055.md`

