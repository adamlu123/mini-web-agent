# CUAWright terminal core

The default persistent Actor orchestrator exposes one model-visible tool:
`run_command`. Ordinary commands execute in the active benchmark environment.
Standalone invocations of the exact Python clients in
`/opt/cuawright-tools` execute in the active environment and are then handled
by the trusted host broker.

Its package boundaries follow the slim Agent/Model/Environment organization
popularized by
[mini-swe-agent](https://github.com/swe-agent/mini-swe-agent). CUAWright keeps
the linear Agent small while placing provider transport under `models/`,
benchmark execution under `environments/`, and explicit composition under
`run/`.

## Preserved behavior

The terminal path derives its Actor, Binder, Verifier, and compaction
prompts from the complete active production prompts. Localized substitutions
replace native image, user-question, specialist, and submission tools with
terminal commands. Task instructions, shared budget reminders, phase
transitions, and specialist requests remain dynamic user messages.

Each benchmark condition selects one immutable named prompt bundle at its
composition root. OSWorld default and `--no-subagents`, WeaveBench, BenchCAD
full/direct, and CADGenBench remain independently named; the prompt registry
does not fall back across benchmark boundaries. The stable CLI exposes
`--no-subagents` only for OSWorld. WeaveBench, BenchCAD, and CADGenBench reject
it, while BenchCAD keeps `full` and `direct` as its benchmark-specific
condition.

OSWorld `--no-subagents` now defaults to the inventory-first linear Actor:
the task instruction is the central requirement, environment inventory is
task-contextual, and the Actor uses direct execution plus final readback. The
orchestrated OSWorld default and explicit `e-l1-direct` / `e-l1-disciplined`
profiles remain unchanged.

`BenchmarkOverlay` remains only as a compatibility adapter for older callers.
Provider-specific suffixes are appended after bundle rendering.

The model still receives:

- the original task and benchmark-specific operating guidance;
- one persistent main Actor, a fresh evaluator-aligned Binder, and a fresh
  independent Verifier;
- the same shared model-call budget and explicit Actor compaction;
- command stdout/stderr, terminal-requested image attachments, and user answers;
- direct Binder report content and independent live-state verification.

The model produces the same work and role reports through six real,
standard-library clients:

```bash
python /opt/cuawright-tools/binder.py --request "TEXT" --max-steps N
python /opt/cuawright-tools/verifier.py --request "TEXT" --max-steps N
python /opt/cuawright-tools/image_show.py --path PATH
python /opt/cuawright-tools/ask_user.py --question "TEXT"
python /opt/cuawright-tools/submit.py
```

Active specialists finish by returning one JSON object containing `status`,
`report`, and `unresolved`; this is ordinary assistant text, not another tool
call. Actor and Binder may use `ask_user.py`; the read-only Verifier may not.
`report.py` is not installed.

Each client is root-owned and non-writable in supported environments. The
broker recognizes only the approved Python executable, exact absolute path,
operation-specific inline arguments, and matching bounded JSON output after the
script actually runs. Shell chaining, redirection, substitution, similarly
named scripts, and the removed `agent ...` protocol do not receive privileges.
Specialist request text is bounded to 2,000 characters; user questions are
nonblank and bounded to 2,000 characters.
The model does not create or name request, question, report, or envelope files.
The host retains immutable request/report artifacts for audit and provenance.

The Actor explicitly chooses each specialist's `--max-steps`. Binder may be
called once and Verifier at most three times. The clients and broker validate
Binder/Verifier step limits of 40/24 while preserving one Actor return call.
The broker captures the final specialist response, validates the native
status/report/unresolved contract, persists immutable host artifacts, and
returns a command result containing the native-shaped report. Verifier receives
the original instruction and live state without Binder conclusions.

`image_show.py` decodes and normalizes the first image frame to PNG before
returning a typed `ImageAttachment`; provider adapters translate that attachment
into actual OpenAI or Anthropic image content rather than printing base64. Host
`TerminalControlState` is authoritative and is not mirrored into model-visible
guest state.

Normal submission requires accepted Binder and Verifier reports with unchanged
report hashes. A latest Verifier status of `repair` blocks submission; `unknown`
preserves uncertainty without authorizing speculative mutation. Ordinary Actor
commands do not mutate control state.

Finalization begins with
`max(16, min(48, round(total_steps * 0.16)))` shared calls remaining. As in the
native orchestrator, the Actor receives one soft reminder but remains free to
use any valid operation. Broker and client errors remain recoverable. Only an
explicit `submit.py` records a normal submission; loop or budget exhaustion is
recorded with forced provenance.

On the final allocated specialist call, no tools are exposed and the specialist
must return its report object. The host captures that response and returns it
directly in the stdout payload of the Actor's still-pending `binder.py`,
or `verifier.py` command. Valid no-tool reports may also complete
early. A generic
blocked/unknown fallback is generated only when no bounded final response or
compatible report exists. Malformed but bounded final text is preserved
verbatim in a forced blocked/unknown report instead of being discarded. This is
the terminal-only equivalent of the native provider-forced
`submit_subagent_report` final call.

## Packages

| Package | Runtime |
| --- | --- |
| `cuawright.environments.osworld` | Maintained wrapper/lifecycle over the upstream OSWorld `DesktopEnv` VM and evaluator |
| `cuawright.environments.benchcad` | Isolated BenchCAD container, `benchcad-run`, `benchcad-render`, and trusted local scoring |
| `cuawright.environments.cadgenbench` | Isolated CADGenBench container and checkpoint/public-evaluation pipeline |

BenchCAD was ported from `benchcad-integration@3c481707`. CADGenBench was
ported from `cadgenbench-integration@1a914bd5`. Their source prompts remain
available verbatim; terminal prompt bundles apply only audited interface
substitutions.

## Build123d-MCP

CADGenBench retains its existing persistent Build123d-MCP service:

```text
run_command -> cad-mcp CLI -> http://127.0.0.1:8765/mcp
            -> container-local Build123d/OCCT session
```

The service is not model-visible as a native tool and is not exposed outside
the isolated container. The model controls `tools`, `schema`, `resources`,
`call`, and `execute` only through `cad-mcp` terminal commands.

## Provider boundary

The normalized adapters preserve OpenAI Responses, direct Anthropic Messages,
and OpenRouter Messages behavior. GPT validation through PhyAGI retains
`session_id`, `strict_session=true`, and configurable model infrastructure
(normally `aiservices`, with alternatives such as `trapi` selected explicitly).
Anthropic retains exact request construction,
strict-schema conversion, caching, compaction accounting, image-history
pruning, payload preflight, and direct/OpenRouter cost tracking. Foundry Fable
threads permit up to ten fully traced policy-feedback continuations. Each
blocked assistant turn is excluded from replay history, and the next user
message reports the policy error while requiring a benign, compliant approach.
Another block after the tenth continuation fails normally.

OSWorld proxy use remains task-driven. A task that requests proxying fails
before VM or model work when its configured proxy still contains example
credentials, exposes credential-bearing configuration to other users, or
cannot complete a host-side HTTPS probe. Proxy-enabled Chrome bypasses
localhost and the configured private website suffix. Operators may record an
explicit `--disable-task-proxy` override when direct guest egress is the
intended condition; tasks that do not request a proxy remain unchanged.

## Launchers

```bash
uv run cuawright-task --task-id 001 \
  --model-provider openai --model gpt-5.5 --reasoning-effort xhigh

uv run cuawright-benchcad \
  --manifest MANIFEST --record-id RECORD_ID --mode full

uv run cuawright-cadgenbench \
  --dataset-dir DATASET --fixture-id FIXTURE
```

CUAWright has one active orchestration path. The deprecated fixed pipeline and
native model-tool path were removed after the terminal path reached production
parity.

## Five-minute health gate

Every paid validation process is inspected five minutes after launch:

```bash
python scripts/python/check_terminal_run_health.py \
  --benchmark osworld \
  --run-dir RUN_DIR \
  --pid PROCESS_ID \
  --process-log PROCESS_LOG
```

The checker writes `health-check-5m.json`. It observes the sole provider tool
(including final no-tool specialist report requests), PhyAGI routing/session
fields, the exact six-client source manifest, exact active client paths,
native-shaped direct report returns, prior-report injection evidence, and typed
image attachment translation when those operations occur. It does not require
removed guest control-file paths or obsolete report-client events. Active
legacy `agent ...` commands, malformed client paths, unavailable clients, dead
infrastructure, or authentication failures are objectively invalid. Slow but
healthy setup or model reasoning is not stopped.
