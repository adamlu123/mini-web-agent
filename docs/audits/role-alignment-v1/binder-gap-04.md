# Binder visual gaps: tasks 051, 058, 060, 061

## Audit summary

The four tasks do **not** support a single “evaluator-aligned Binder” rule.
Binder should describe the task-visible, native artifact and require direct
final-state proof; it must not infer hidden evaluator predicates.

- **051:** Binder/Critic described the intended editable visual composite, but
  the evaluator mainly scored XCF layer names, bounding boxes, IoU, and SIFT
  color proxies. Frames, overlap, editability, preservation, and no-crop
  semantics were largely unscored. The Critic’s hash-preservation patch was
  evaluator-invisible.
- **058:** Binder/Critic alignment improved after the visible tutorial and exact
  3-D values became identifiable. However, both attempts were
  infrastructure-aborted before artifact creation; the missing-file zero must
  not be attributed to the Actor. The evaluator hard-gates the PPTX and then
  combines OOXML structure, Morph, image SIFT/RANSAC, and exact camera checks.
- **060:** Binder’s one-day-per-slide semantics conflicted with an evaluator
  expecting a 28-column, four-day strip and incorrectly repeating all 18
  hyperlink checks on every slide. The artifact had correct 5/5/6/2 events and
  links; missing Morph was the genuine Actor defect. Critic’s concern about
  categories was based on visible-source absence, while the hidden evaluator
  workbook supplied them—this must remain unresolved for Binder.
- **061:** Binder and Critic were fully aligned, but their semantic contract
  exceeded the evaluator. The evaluator checked only output existence,
  decoding, shape, and pixel MSE against a hidden target. It did not test source
  preservation, scene identity, watermarks, EXIF, or explicit style semantics.

## What Binder can safely require

1. **Canonical final artifact:** exact path, filename, format, and basic
   dimensions when stated or visibly verifiable.
2. **Native persistent structure:** reopenable XCF/PPTX/JPEG state; named
   editable layers/shapes, hyperlinks, transitions, or other native objects
   when the task requests them.
3. **Visible semantic content:** composition, subject identity, frame/overlay
   appearance, per-day event placement, venue text, image style, etc.,
   supported by supplied task/reference material.
4. **Source preservation:** byte/hash verification when originals are
   available and preservation is requested.
5. **Final proof:** save, close/reopen or parser readback, inspect the canonical
   path, verify metadata/hashes, and provide a rendered/slideshow/image
   comparison for visual claims—not merely a successful script or screenshot.

## What cannot be generalized

- Do not treat layer bounding boxes, IoU, SIFT, object counts, or raw MSE as
  universal substitutes for visual semantics.
- Do not require hidden evaluator data, hidden reference files,
  evaluator-cache artifacts, or undocumented layout assumptions.
- Do not invent exact tutorial IDs, camera values, animation mechanisms, or
  operation history when the visible source is ambiguous.
- Do not generalize “one native operation,” exact editor history, or hidden
  pixel equality.
- Do not interpret a low score as proof of semantic failure where the evaluator
  is proxy-based, or interpret zero as Actor failure when infrastructure
  prevented submission.

## Conclusion

Binder should require evidence-backed native final state plus task-visible
visual semantics, while explicitly separating unresolved facts from
evaluator-only behavior.

## Evidence references from the audit

- `reliability64-alignment-audit/task-051.md`: lines 13–26 and 48–57
- `reliability64-alignment-audit/task-058.md`: lines 14–22, 24–34, and 36–51
- `reliability64-alignment-audit/task-060.md`: lines 5–11, 19–25, and 29–38
- `reliability64-alignment-audit/task-061.md`: lines 4–23 and 27–45
- `/home/v-theolee/osworld-harness-handoff.md`: canonical native/readback rules
