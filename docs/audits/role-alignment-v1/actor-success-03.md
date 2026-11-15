# Actor native document controls: tasks 003, 066, 091, 107

## Complete-state pattern

```text
exact contract
-> native/editable mutation
-> save/reload
-> parser/API/hash/render readback
-> exact-path handoff
```

## Task 003

Actor used bounded image construction, native PPTX background XML/relationships,
pixel/hash checks, slide-tree preservation, and rendered validation.

## Task 066

Actor rendered the untouched deck, parsed native geometry, made a narrow atomic
OOXML edit, corrected coordinates, rerendered, and verified native object
identity/no duplicates.

## Task 091

Broad native construction succeeded only through repeated parser checks,
render/contact-sheet feedback, geometry fixes, stale-text scans, and reopening
WPS to avoid stale in-memory overwrite.

## Task 107

Textually inserted PCB segments were not recognized as native tracks. Audit
exposed the failure; repair switched to KiCad’s native `pcbnew` API, rebuilt
connectivity, saved/reloaded, and verified the archive.

## Related failures lacked

- short feedback loops;
- native parser readback;
- render validation;
- evaluator-facing handoff identity;
- resistance to premature submission.

## Evidence

- selected task traces/evaluation results for 003, 066, 091, 107
- `docs/CLI_HARNESS_PLANNING_FRONTIER_ANALYSIS.md`
- `/home/v-theolee/osworld-harness-handoff.md`

