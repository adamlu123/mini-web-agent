# Telemetry extraction method

## Scope and ordering

The bundle covers every immediate attempt directory in the complete Direct and
Full result roots: 213 Direct attempts and 213 Full attempts. Selection is read
from each condition's `batch/selected-runs.tsv`; only attempt basenames are
emitted. Rows are sorted by condition, record ID, and attempt basename. HTTP
rows are sorted by condition, record ID, and log basename.

## Deterministic extraction

- `attempt-telemetry.jsonl` reads response metadata from Direct
  `responses/response.json` and Full `responses/step-*.json.gz`. It sums only
  `status` and `usage` fields; response output, reasoning, prompts, and tool
  content are never copied.
- Full successful role counts come from `model_step` rows in
  `trace-slim.jsonl`. Direct has one `direct` role per successful response.
  The six terminal policy calls have no response JSON; each is counted
  separately as one failed Actor call from the active Actor stage and the
  terminal HTTP 400/policy classification.
- Input, cached input, output, and reasoning tokens are recorded API usage.
  Uncached input is `input - cached`; visible output is `output - reasoning`.
  Reasoning tokens remain included in output and are not double-counted.
- Per-attempt estimated cost is
  `uncached_input * $5/M + cached_input * $0.50/M + output * $30/M`, rounded to
  nine decimal places. Aggregate cost is recomputed once from aggregate token
  integers using the same recorded pricing, preserving the authoritative
  floating-point result.
- `http-telemetry.jsonl` extracts every `HTTP/1.1 NNN` status from each batch
  log. `retryable_status_attempts` counts 429/502/503/504 responses, matching
  the final report's retry accounting. A log is marked selected when it
  contains the selected attempt; notes identify batch retries, terminal policy
  400s, transient HTTP statuses, and the Full log spanning the recovered
  infrastructure attempt.
- Direct attempt timing comes from `batch/status.tsv`, except the selected
  hinge retry, which comes from `batch/retry-status.tsv`. Full selected timing
  normally comes from `batch/status.tsv`. The recovered Full failed attempt is
  bounded by `manifest.json` start and its trace `infrastructure_failure`;
  the selected recovery is bounded by its manifest start and batch end.
- Selected terminal classifications come from the final record evidence.
  The two unselected attempts are classified from their terminal evidence as
  `gateway_failure` and `infrastructure_failure`.
- `orchestration-telemetry.jsonl` records the recovered Full task's 2.698368s
  batch-start-to-first-attempt interval and 1.082088s inter-attempt retry gap.
  These 3.780456s are present in the authoritative batch wall-clock duration
  but are not attributable to either attempt directory.
- `telemetry-summary.json` independently sums the emitted attempt accounting
  and HTTP rows, recomputes selected timing distributions with linear
  percentile interpolation, and fails generation unless calls, roles, tokens,
  costs, response/HTTP/status counts, retries, and timing equal `summary.json`.

## Limitations

- HTTP telemetry is log-scoped. The recovered Full record has one batch log
  spanning two attempt directories, so its HTTP statuses cannot be assigned
  uniquely between the failed infrastructure attempt and selected recovery.
- Batch TSV timestamps have one-second resolution. Full attempt manifests and
  traces have finer resolution, but final-report timing intentionally uses the
  batch TSV wall intervals. Attempt durations plus the separately recorded
  orchestration intervals reconcile that total exactly.
- Direct's failed, unselected hinge attempt has ten upstream 504 responses and
  no successful response JSON, so it contributes no calls or tokens.
- A successful API response may have terminal response status `incomplete`;
  these three Direct responses remain successful calls exactly as recorded.
- Costs are list-price estimates, not billed amounts. No raw responses,
  reasoning, prompts, command output, credentials, or absolute run paths are
  included.
