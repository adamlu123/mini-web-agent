# Actor native geometry strategy: tasks 049, 051, 076, 087, 090, 103, 108

## Finding

Actors often commit the wrong coordinate model first, then serialize it
consistently. A valid native file, plausible render, and self-check can all pass
while geometry fails.

## Failure patterns

- **049:** arbitrary connector offsets instead of box-center anchors.
- **051:** full-canvas proxy layers destroyed native XCF bounds.
- **076:** semantic grid formulas differed from actual absolute geometry.
- **087:** native objects existed but animation semantics/count/order were
  incomplete.
- **090:** structurally valid lookalike lacked required visual/native
  relationships.
- **103:** coherent CAD generator used guessed dimensions and lacked analytic
  fillet surfaces.
- **108:** identities/nets existed, but placement, rotation, routing, DRC, and
  archive handoff remained incomplete.

## Productive strategy conditions

1. Evaluator reads the same structured artifact being edited.
2. Geometry is derived from visible dimensions/object bounds/reference pixels.
3. Independent post-save parser/geometry/connectivity readback is performed.
4. Render is visual feedback, not proof of native correctness.
5. Serialization/archive occurs only after geometry/connectivity checks pass.

## Complete native state

- canonical path and complete package;
- real editable objects, not padded rasters/lookalikes;
- serialized coordinates, rotations, dimensions, IDs, timing, and surfaces;
- cross-object relationships and connectivity;
- parser/render/DRC/readback after save;
- preserved unrelated state;
- archive identity with the final project.

## Evidence

- selected task traces/results for 049, 051, 076, 087, 090, 103, 108
- corresponding evaluator implementations
- `/home/v-theolee/osworld-harness-handoff.md`

