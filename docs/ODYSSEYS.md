# Odysseys

CUAWright treats Odysseys as an external benchmark identity, not as vendored
source and not as an implicitly “public” checkout. Setup requires the exact
source, dataset, generated examples, judge script, desktop runtime, VM, and
credential paths.

## Source identity

`cuawright setup odysseys` accepts a full 40-character source commit. The
checkout must be clean, the observed commit must match, and the configured
dataset, examples, and judge must be inside that checkout. Setup records the
dataset and judge SHA-256 values. Doctor and every run recheck the commit,
cleanliness, paths, and hashes.

The desktop runtime is a separate pinned external dependency at
`43997ed9a88b81151bbb496f1fd1f3b9febf312c`; only its DesktopEnv/provider
surface is used. It must also be clean, including untracked files. The launcher
temporarily prepends that checkout, verifies every loaded `desktop_env` module
origin, disables bytecode writes while it is active, serializes concurrent
activations, and restores the prior import state after the environment closes.
CUAWright does not import an Odysseys package or evaluator into the agent loop.

## Run

```bash
cuawright run odysseys \
  --task 440ed7f388a2a4528a8d9fb75f83e11f934b5b5d \
  --result-root results/odysseys
```

The public profile is GPT-5.5 with xhigh reasoning through the existing OpenAI
provider factory and the sole model-visible `run_command` tool. OpenAI has no
enforceable local dollar tracker, so the internal launcher rejects
`--max-dollars` for OpenAI rather than silently ignoring it. Providers with
current factory-enforced tracking retain their normal cost behavior.
Odysseys desktop runs explicitly disable proxy setup; a future task that requests
it fails closed rather than consuming an unverified or empty proxy pool.

## Complete trajectory and finality

After every dispatched live-environment command, the adapter captures a fresh
desktop screenshot and a trajectory row containing role and stage. This
includes:

- Actor ordinary commands;
- the Actor command that invokes a specialist;
- every active specialist's ordinary commands (currently Binder and Verifier;
  Critic is also covered if enabled by a future current runtime); and
- accepted normal or host-classified forced submission.

A guest `submit.py` exit is not final. A row becomes final only after the
trusted broker has persisted an accepted submission and returns
`completed=True`. Normal submission also requires a safe, nonempty UTF-8
`/home/user/.agent/odysseys/final_response.md`. Forced submission is retained
with an explicit `forced` classification. Finalization reconciles the recorded
action count, consecutive trajectory rows, and canonical screenshot hash
entries one-to-one. Missing, duplicate, orphaned, or unsafe screenshot
references make the run incomplete.

## External judging

CUAWright never starts the paid image judge automatically. Each run copies the
exact judge script, complete task source, selected task record, and generated
example into `odysseys-judge/TASK_ID/inputs/`. The copied inputs are made
read-only and all judge inputs, trajectory rows, screenshots, configuration,
sentinel, and final response are hashed in `judge-input.json`.

Use the emitted `judge_command`. Its CUAWright wrapper verifies every recorded
hash and a separately supplied manifest digest immediately before replacing
the mutable run inputs with a complete private read-only snapshot. It invokes
the snapshotted judge against only that private config, trajectory, screenshots,
task source, and result sentinel. It derives the model and all paths itself; the
manifest cannot supply Python arguments. Open descriptors pin the private
snapshot across parent-directory replacement, and successful judge output is
written through an inherited anonymous regular-file descriptor. After complete
process-group cleanup, the wrapper seals that descriptor against writes, growth,
and truncation before reading or validating its exact bytes. Publication through
the originally opened run directory occurs only after exit code zero and
validation as a nonempty JSON object or list. Nonzero, mutable, oversized, or
malformed partial output leaves any prior canonical result untouched. Any live
mutation after finalization fails closed.

The numeric `result.txt` is only the upstream completed-run sentinel. It is not
a score. The paid judge writes rubric results separately.

## Batch

`python -m cuawright.run.odysseys_batch` is a thin bounded launcher over
`cuawright.operations.scheduler.run_process_jobs`. It records source identity,
per-task logs/PIDs/exits, selected run and result hashes, and separate counts for
ready, incomplete, failed, interrupted, and missing tasks. Readiness requires a
completed scheduler outcome with exit code zero; an interrupted zero exit is
never accepted. A neutral process monitor invokes the current registry-backed
health checker for every task after five minutes or earlier process completion,
without polling or reaping the process leader. Descendant discovery is completed
before leader collection, and late owned descendants are identity-bound and
cleaned before a scheduler outcome can be accepted. The batch retains both
task-local and batch health artifacts. A new workload remains behind an
inherited launch gate until its leader pidfd and start identity are registered;
registration or gate-release failure kills and reaps the still-gated launcher
and records its actual PID and wait result. The batch does not execute the
external judge.

## PR #26 fidelity matrix

Labels: **architecture-only** changes implementation boundaries without
intending benchmark-visible change; **bugfix/safety** preserves the intended
outcome while failing closed; **deliberate semantic divergence** records an
intentional behavior difference required by the validated current architecture
or complete provenance.

| Area | PR #26 intended behavior | Current port behavior | Classification and reason |
| --- | --- | --- | --- |
| Task loading/source match | Unique 40-character task ID; task instruction and start website must match the generated OSWorld example; record dataset/example hashes | Same task checks, plus a clean explicit source commit, configured-path containment, repeated hash checks, and immutable selected inputs | **Bugfix/safety:** task semantics preserved; arbitrary or mutable checkout identity is rejected |
| Shared call budget | 100 calls shared by Actor, specialists, and compaction | Current `SharedStepBudget(100)` is shared by every invoked Actor/specialist/compaction response | **Architecture-only:** current budget implementation replaces the PR-local composition |
| Model/tool profile | GPT-5.5, xhigh reasoning, sole model-visible `run_command` | Same public profile through the current provider factory and TaskRunner | **Architecture-only** |
| Chrome/CDP overlay | Reuse setup-opened Chrome on `DISPLAY=:0`, CDP 1337/9222, and the OSW Playwright interpreter; keep useful evidence foregrounded | Preserved in the independent `odysseys-default` bundle with pinned hashes | **Architecture-only:** adapted to current prompt layers without changing OSWorld or WeaveBench |
| Screenshot chronology | Capture a fresh screenshot after each main-Actor task/evidence command | Capture after every broker-dispatched Actor or specialist command, in dispatch order | **Deliberate semantic divergence:** complete live-environment provenance replaces Actor-only omission |
| Actor/specialist roles | Binder, Critic, and Verifier existed, but their invocation/internal commands were excluded from `traj.jsonl` | Rows retain Actor invocation plus ordinary specialist commands with role/stage. Current production topology activates Binder and Verifier; Critic capture remains supported if the current runtime enables it | **Deliberate semantic divergence:** uses the validated current topology and does not misattribute specialist actions |
| Final response | Require safe, nonempty UTF-8 `.agent/odysseys/final_response.md` before normal submission | Same requirement; copied into immutable judge artifacts | **Bugfix/safety:** stronger secure-read and artifact validation only |
| Submission finality | A successful guest `submit.py` execution made the Actor row final | Final only after the trusted broker persists acceptance and returns `completed=True`; budget-forced submission is explicitly `forced` | **Bugfix/safety:** guest exit alone is insufficient |
| Trajectory/screenshot integrity | Emit judge-compatible trajectory rows and mirrored screenshots | Additionally require consecutive action counts and exact one-to-one canonical screenshot references/hashes, with no missing, duplicate, orphaned, or unsafe entries | **Bugfix/safety** |
| `result.txt` | Write numeric `0.0` only as the upstream completed-run sentinel, not a score | Same | **Architecture-only** |
| External judge command/inputs | Emit a separate command using the external script and live task source; do not start it | Emit a digest-bound wrapper; retain exact judge/task bytes, verify them, build a private read-only snapshot, run fixed arguments, and publish only valid successful output | **Bugfix/safety:** paid judge remains external while mutable-input and argv injection paths fail closed |
| Batch concurrency/retries/cleanup | Bounded width and dataset-order/explicit selection; each task gets the runner's one infrastructure-only fresh-VM retry; coordinator tracks logs, health, exits, and summaries | Bounded public process scheduler, same selection and per-task infrastructure retry/no quality retry, PID identity cleanup, automatic per-task checks through the current factual health registry, and explicit ready/incomplete/failed/interrupted/missing accounting | **Architecture-only** for scheduler/health implementation; **bugfix/safety** for cleanup and accounting. OpenAI dollar limits are rejected because the current factory cannot enforce them |
| Automatic paid judging | Never invoke the rubric judge during task or batch execution | Same | **Architecture-only** |

No intended task, Chrome, response, sentinel, retry, health, or paid-judge
boundary is silently substituted. Explicit semantic differences are identified
per row rather than being presented as byte-equivalent PR behavior.

## Bounded prompt comparison and integration decision

PR #26 did not provide a score, controlled A/B run, web-benchmark comparison, or
other measured evidence that its Odysseys wording improves general web-agent
performance. Its docs describe the adapter contract, while the commit message
reports offline test counts. Neither reports an Odysseys, OSWorld, WeaveBench,
or other web-benchmark result attributable to the prompt.

The comparison below is intentionally bounded to the model-visible additions
that could be mistaken for general web-agent tuning:

| PR #26 wording/behavior | Current OSWorld bundle | Current WeaveBench bundle | Port decision |
| --- | --- | --- | --- |
| “Work only through the live public web and the visible Chrome profile” and no evaluator/rubric/reference/private API access | Already says to preserve the setup-opened application, website, account, profile, and task state; forbids hidden backend/private API state | Already requires the real desktop and genuine GUI actions, but is not necessarily a Chrome task | Keep the evaluator/rubric prohibition Odysseys-only; the general live-state rule is already covered |
| Fixed `DISPLAY=:0`, CDP `1337` plus `9222`, the OSW Playwright interpreter, `connect_over_cdp`, and no package/port rediscovery | Already names the setup-opened Chrome session, interpreter, `1337`, page selection by URL/title, and recovery when needed; it does not promise `9222` for every task | Has real-display GUI instructions but no Chrome/CDP contract | Keep exact endpoints and interpreter procedure Odysseys-only because they are environment facts, not generic policy |
| “bring_to_front”, wait for rendering, real visible navigation/clicks, and avoid background fetches, hidden tabs, headless requests, or bulk `page.content()` | Already prefers accessibility locators, visible task state, and forbids hidden backend substitution; it is less prescriptive about foreground/render timing | Already requires iterative visible GUI action and real screenshots; it has no per-command browser-page rule | Preserve for Odysseys judge evidence. The stronger generic foreground/render wording is only a candidate for an isolated future experiment |
| A saved screenshot is the rubric-credit surface after every judged command | No per-command image-judge contract | Screenshots support observation and anti-fabrication, not an Odysseys trajectory judge | Odysseys-only and required |
| Exact response at `.agent/odysseys/final_response.md`, with normal submit rejected until present | Uses broker submission artifacts, not this response file | Uses benchmark outputs under `/tmp_workspace/`, not this response file | Odysseys-only and required |
| PR excluded specialist invocation/internal commands from the judged trajectory | No Odysseys trajectory | No Odysseys trajectory | Intentionally changed: capture every dispatched Actor/specialist command with role/stage so provenance is complete |

Accordingly, this port preserves the PR's concrete Odysseys Chrome procedure,
foreground evidence discipline, screenshot-credit explanation, and canonical
final-response requirement in `odysseys-default`. It does **not** alter any byte
of the OSWorld or WeaveBench bundles. Any future cross-benchmark trial of the
stronger foreground/render wording must use an isolated named experimental
bundle, retain exact before/after prompt hashes, and obtain byte-preservation
approval before changing a production bundle.
