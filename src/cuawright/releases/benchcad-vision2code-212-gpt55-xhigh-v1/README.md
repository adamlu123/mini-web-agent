# BenchCAD 212-record evidence bundle

Compact evidence for `docs/CLI_HARNESS_BENCHCAD_212_RESULT.md`.

- `panel.jsonl`: frozen IDs, strata, and input/GT hashes.
- `records.jsonl`: record-level scores, statuses, and prediction hashes.
- `families.tsv`: family-level results.
- `direct-submission.jsonl` / `full-submission.jsonl`: exact regradable predictions.
- `selection.json`: deterministic selection and benchmark pins.
- `summary.json`: authoritative aggregate arithmetic and usage.
- `provenance.json`: source artifact and verification hashes.
- `runtime-source-sha256.txt`: source hashes captured before paid execution.
- `attempt-telemetry.jsonl` / `http-telemetry.jsonl`: recomputable usage,
  timing, role, retry, and terminal-outcome evidence.
- `orchestration-telemetry.jsonl`: batch/retry intervals not attributable to an
  attempt directory.
- `telemetry-summary.json` / `telemetry-method.md`: exact reconciliation and
  extraction method.
- `image-inspect.json` / `image-history.jsonl` / `image-archive.json`: executed
  OCI image identity and local archive provenance.

The banking commit contains a post-run missing-submission provenance fix. Exact
executed source hashes remain in `runtime-source-sha256.txt`; the fix did not
affect any scored prediction in this result.
- `attempt-telemetry.jsonl`: per-attempt calls, roles, tokens, cost, timing,
  response status, selection, and terminal classification.
- `http-telemetry.jsonl`: per-log HTTP statuses and retry/selection notes.
- `telemetry-summary.json`: exact aggregate telemetry reconciliation.
- `telemetry-method.md`: deterministic extraction method and limitations.
