# CLI Harness Architecture

The CLI harness runs OSWorld V2 tasks through Binder, Critic, Actor, and fresh
Verifier roles while exposing only Bash inside one initialized Ubuntu VM.

The validated default profile is pinned by tag `cli-harness-v2-55pct-106`.
Measured score, cost, provenance, and known failed-task causes are recorded in
`docs/CLI_HARNESS_55PCT_RESULT.md`.

## Lifecycle

1. Start a Docker-backed OSWorld VM and run the selected task setup.
2. Provision the guest-only `osw` helper and its small knowledge base.
3. Run Binder to draft bindings and retain its evidence in the trace.
4. Run a fresh Critic thread against the draft and the same initialized VM.
5. Resume Binder to adjudicate any provisional patches or unresolved probes.
6. Run Actor in the same VM using the final authoritative bindings.
7. Run a fresh read-only Verifier against the original task, final bindings,
   unresolved Critic probes, and live VM.
8. If the audit finds directly evidenced failures, resume the original Actor
   thread for one bounded repair round.
9. After Actor repair, run a new fresh Verifier recheck.
10. Run the task's real `env.evaluate()` implementation without resetting.

The hard maximum is 300 model steps. Binder is capped at 40, Critic at 32,
Binder adjudication at 16, and 52 steps are protected for post-Actor handling.
Actor therefore receives at least 160 steps when every planning ceiling is
exhausted, and unused planning steps expand its initial allowance. Fresh audit
uses at most 24 steps, resumed Actor repair at most 32, and fresh recheck at
most 8. Actual role usage is charged against the same global maximum. A
malformed final Binder or adjudication `submit_bindings` call receives one
serialization-only retry and is charged normally.

Binder emits one thin binding per independently scoreable acceptance condition:

```json
{
  "id": "editable-copy",
  "description": "Save an editable image with separate text and photo layers.",
  "critical": "/home/user/Desktop/final.xcf",
  "prove": ["Open the XCF and inspect its layer structure."],
  "source": "Save an editable copy."
}
```

Bindings describe requirements and proof, not evaluator internals, guessed
weights, or execution plans.

## Model tools

- `run_command`: executes ordinary Bash through the guest `/execute` endpoint,
  wrapped by GNU timeout with process-group TERM/KILL cleanup.
- `read_image`: fetches an image from a VM path and attaches it to the next
  Responses API request.
- `submit_bindings`: ends Binder with the binding list.
- `propose_binding_changes`: records evidence-backed provisional Critic patches
  or unresolved probes against a candidate copy.
- `withdraw_binding_changes`: removes a Critic patch disproven by counterevidence.
- `submit_critique`: ends Critic with a concise summary; Binder remains final.
- `submit_task`: ends Actor and begins independent verification.
- `submit_verification`: records one fresh `pass`, `fail`, or `unknown` verdict
  per binding plus any original-task coverage gaps.

Actor does not self-certify completion. The Verifier receives no Actor ledger or
Actor transcript. Audit mode is inspection-only. Directly evidenced failed
bindings are returned to the original Actor session; passing and unknown
findings are not repair items. Actor does not certify its repair: a new fresh
Verifier rechecks the live state. Multi-phase tasks currently bypass post-Actor
verification and retain their native phase-local execution/evaluation flow.

Actor has one execution pool rather than rigid substages. The harness injects
soft minimum-deliverable and finalization reminders with the remaining call
count, but does not delay early submission or require a completion ledger.

If the Actor needs information only the user can provide, it asks a plain-text
question. The harness routes that question through the task user simulator when
configured, otherwise through OSWorld's standard no-additional-information
response.

`osw "search terms"` is available on the guest `PATH` only as an FTS5/BM25
knowledge-base search engine. It returns short procedural excerpts. Agents use
normal Bash, Python, and installed applications directly for screenshots,
AT-SPI, Chrome CDP/Playwright, and all task actions.

Native application state is preferred when it is part of the task. Chrome work
should use Playwright attached to the existing CDP endpoint and favor
accessibility roles over brittle DOM selectors. Direct artifact manipulation is
appropriate when the evaluated output is fundamentally a file. Externally
edited artifacts should be reopened in the native application when stale state,
file locks, or application-visible state may matter.

## Responses API state

- Model: `gpt-5.5` by default.
- Normal Actor calls: 16,384 maximum output tokens.
- Binder calls: 16,384 maximum output tokens.
- Actor compaction calls: 16,384 maximum output tokens.
- Verifier audit and recheck calls: 16,384 maximum output tokens.
- Reasoning effort: medium by default.
- Encrypted reasoning items are replayed with complete normalized context.
- `session_id` changes between Binder, Actor, and each compacted Actor epoch.
- `strict_session` is enabled through the gateway.

After 40 Actor tool calls, the current thread produces a textual handoff. A new
Actor thread starts from the task instruction, bindings, and handoff.

## Traces

Each run writes:

- `trace-slim.jsonl`: reasoning summaries, tool names/arguments, output sizes
  and hashes, bindings, compactions, submissions, and evaluation results.
  Command output bodies are excluded.
- `trace.jsonl`: the same events with command output capped at 16,000
  characters.
- `requests/*.json.gz`: complete request input arrays for every model call.
- `responses/*.json.gz`: complete raw responses.
- `command-output/*.txt.gz`: full command results.
- `images/`: images supplied through `read_image`.
- `evaluation/result.json`: the final evaluator payload.
- `verification-audit.json`: the first fresh audit.
- `verification-repair.json`: the resumed Actor repair report, when run.
- `verification-recheck.json`: the fresh post-repair audit, when run.
- `verification.json`: combined audit/repair/recheck accounting.

Truncated command output tells the model to issue a narrower command.

## Infrastructure and concurrency

Benchmark execution defaults to `v2026.06.24` and may explicitly select
`v2026.08.08`. The source commit, task classes, task assets, Docker QCOW2, and
website stack must all match the selected release.

Local task websites use:

```bash
WEBSITE_HOST_SUFFIX=10.217.29.172.nip.io
```

Each concurrent experiment uses its own Docker VM overlay, dynamically allocated
ports, gateway session namespace, run directory, and evaluator cache. The QCOW2
and task assets are shared read-only.

Docker guests default to an 80G sparse virtual disk unless a task explicitly
requests another size. The provider expands the guest partition for both default
and task-specific sizes. After task setup and guest-tool provisioning, the
harness requires at least 5 GiB free on the guest root filesystem.

Agent shell commands use `/execute` with `/bin/bash -c` and an inner GNU timeout.
The server HTTP timeout is longer than the command timeout plus kill grace.
Return status is normalized from `returncode` because `/execute` reports
top-level success even for nonzero commands. Exit codes 124/137 are deterministic
model-visible timeouts, not guest infrastructure failures.

Guest command transport failures are tracked separately from deterministic
command errors. Three consecutive controller failures, or guest disk exhaustion,
abort the run and trigger one clean whole-task retry in a fresh VM. Binder,
Critic, and Actor prompts also require narrow inspection and disk checks before
resource-expansive operations. Retryable custom task setup/control-channel
failures are normalized to the same clean whole-task retry path.

Responses API retries classify typed connection, timeout, rate-limit, and
transient status exceptions rather than relying only on message text. Retry
trace events retain the exception type and bounded cause chain so connection
failures such as a generic `APIConnectionError: Connection error` exhaust the
configured harness backoff instead of terminating after the first retry.
