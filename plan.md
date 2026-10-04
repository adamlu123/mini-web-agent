# Plan: one CLI Actor-orchestrator with web, CadGenBench, and BenchCAD adapters

> Decision: build one persistent, Actor-controlled CLI orchestration runtime and
> attach three task environments to it: mini-web/OM2W, CadGenBench, and BenchCAD.
> Do **not** port or recreate the fixed sequential Binder → Critic → Binder
> adjudication → Actor → audit → repair → recheck pipeline.

The dedicated CAD commands will always select the CLI Actor-orchestrator. They will
not expose an ambiguous `--mode full`, a sequential fallback, or BenchCAD's direct
no-tool agent mode. BenchCAD's offline score command is retained because it is trusted
post-processing, not another agent topology.

## Sources pinned for the port

- Actor-orchestrator contract: `/home/luyadong/sandbox/deskwright-cua/AGENTS.md`,
  behavior `dbaf2a54`, AIServices routing `e727ca43`, and tag
  `cli-harness-v2-60pct-106-repro-v1`.
- [CadGenBench integration branch](https://github.com/TheoLeeCJ/deskwright-cua/tree/cadgenbench-integration/cadgenbench_harness):
  local ref `origin/cadgenbench-integration` at `1a914bd51e29193bdc4c3a710c981f5670fe13f9`.
- [BenchCAD integration branch](https://github.com/TheoLeeCJ/deskwright-cua/tree/benchcad-integration/benchcad_harness):
  local ref `origin/benchcad-integration` at `3c481707753f35b06118492bdc635bbb7199b760`;
  banked implementation/result tag at `fa1a6a09`.

Do not cherry-pick either benchmark branch wholesale. Their merge bases predate the
current DeskWright runner, and each branch edits a large, benchmark-conditional
`cli_harness/runner.py`. Port the isolated task, container, validation, packaging,
and scoring code; re-home orchestration in mini-web's new common runtime.

Execution-flow diagram: [high-resolution PNG](docs/cli_actor_orchestrator_execution_flow@2x.png),
[SVG wrapper](docs/cli_actor_orchestrator_execution_flow.svg), and
[editable HTML source](docs/cli_actor_orchestrator_execution_flow.html).

## Target commands

```bash
mini-web \
  --task-id <task-id> \
  -c generation/cli_actor_orchestrator_gpt55.yaml \
  --actor-orchestrator \
  --total-steps 100 \
  --compact-every 20
```

```bash
mini-web-om2w \
  --tasks-file src/miniswewebagent/run/benchmarks/om2w_260220.json \
  -c generation/cli_actor_orchestrator_gpt55.yaml \
  --actor-orchestrator \
  --total-steps 100 \
  --compact-every 20 \
  --workers 8
```

```bash
mini-web-cadgenbench run \
  --dataset-dir cache/cadgenbench-data \
  --fixture-id 101 \
  --cadgenbench-repo cache/cadgenbench \
  --image deskwright-cadgenbench:33304cf \
  -c generation/cli_actor_orchestrator_gpt55.yaml \
  --total-steps 150 \
  --compact-every 40
```

```bash
mini-web-benchcad run \
  --manifest cache/benchcad-smoke/manifest.json \
  --record-id washer_000001_s20260505 \
  --benchcad-repo cache/BenchCAD-main \
  --image deskwright-benchcad:52087ef \
  -c generation/cli_actor_orchestrator_gpt55.yaml \
  --total-steps 300 \
  --compact-every 40 \
  --defer-score
```

```text
persistent main Actor
  ├─ run_command
  ├─ read_image
  ├─ call_binder(max_steps)
  ├─ call_critic(max_steps)
  ├─ call_verifier(max_steps)
  └─ submit_task
          │
          └─ benchmark runtime
               ├─ mini-web local workspace/browser CLI
               ├─ CadGenBench container + fixture-local cad-mcp CLI
               └─ BenchCAD container + benchcad-run/render CLI
```

<details>
<summary>1. Comparison and integration decisions</summary>

| Concern | CLI Actor-orchestrator source | CadGenBench branch | BenchCAD branch | mini-web decision |
|---|---|---|---|---|
| Agent topology | Persistent Actor dynamically calls fresh Binder, Critic, and Verifier | Already uses that topology | `full` is the rejected fixed sequential topology | One common dynamic runner; port no fixed stages |
| Model tools | Six stable Responses function tools | Same six tools; CAD stays behind shell | Fixed-role submission tools plus shell | Expose exactly the six Actor tools for every profile |
| Task operations | `run_command` and `read_image` | `cad-mcp` is invoked through `run_command` | `benchcad-run` and `benchcad-render` are invoked through `run_command` | Benchmark commands remain ordinary CLI; no CAD schemas become Responses tools |
| Budget | One `SharedStepBudget` for Actor, roles, and compaction | Hard maximum/default 150 | Fixed-flow run used 300 | Common budget class with profile maximums: web 300, CadGenBench 150, BenchCAD 300 |
| Initial images | Task-dependent | Every declared drawing/render goes to Actor and each fresh specialist | Only `/input/input.png` is model-visible | Runtime supplies immutable initial images to every fresh context |
| Mutation ownership | Main Actor intended mutator; specialist non-mutation is prompt-level | Same | Fixed flow has several framework-owned mutation stages | Main Actor owns mutations; no framework repair stage |
| Evaluation | Benchmark-specific and outside orchestration | Public validity/render only; official score remains external | Trusted voxel-IoU scorer | Keep evaluator data and score feedback outside every model context |
| Failure recovery | Gateway/guest retry classification | Last-valid STEP checkpoints plus MCP/container recovery | Hardened freeze/copy/scorer cleanup | Preserve adapter-specific recovery behind one runtime protocol |
| Existing default | Sequential path exists when source flag is absent | Dedicated runner forces Actor-orchestrator | `--mode full` selects fixed flow | CAD commands have only Actor-orchestrator; ordinary mini-web stays single-agent without its flag |

### Result claims

- CadGenBench's source run reports 81/81 valid and score 0.4979 at GPT-5.5 xhigh.
  This is a reference for behavioral parity, not a score this port may inherit.
- BenchCAD's banked 212-record result reports mean IoU 0.7390 versus 0.4378 for
  its direct condition. That result came from the fixed sequential topology and
  cannot be attributed to the new Actor-orchestrator integration.
- The BenchCAD Actor-orchestrator profile is therefore a new treatment and needs
  its own frozen-manifest evaluation before any performance claim.

### Why selective porting is required

- CadGenBench changes roughly 7,100 lines and BenchCAD roughly 3,100 lines from
  older branch bases, including mutually incompatible edits to the same runner.
- Copying either runner would reintroduce OSWorld-specific infrastructure and, for
  BenchCAD, the exact sequential flow excluded by scope.
- The reusable boundary is the benchmark environment: task discovery, immutable
  inputs, container execution, bounded reads, lifecycle cleanup, provenance,
  validation, packaging, and trusted scoring.

</details>

<details>
<summary>2. Common CLI Actor-orchestrator contract</summary>

### Main Actor

- One persistent Actor owns inspection, implementation, repair, and submission.
- Its tool list and ordering remain stable across turns and compaction:
  `run_command`, `read_image`, `call_binder`, `call_critic`,
  `call_verifier`, `submit_task`.
- It chooses whether and when to invoke each specialist and chooses `max_steps`.
- Specialist reports return as ordinary function-tool outputs. They never cause a
  runner-owned transition or automatic repair.
- The Actor alone calls task-mutating CLI operations by policy and decides whether
  specialist advice is supported by live evidence.

### Specialists

- Each invocation gets a fresh Responses context and unique session id.
- Binder returns a concise acceptance brief, exact visible facts, canonical target
  artifacts, preservation constraints, and unresolved questions.
- Critic challenges the latest Binder report and public/live evidence.
- Verifier inspects the actual canonical candidate and returns direct evidence plus
  prioritized repair guidance.
- Specialists receive `run_command`, `read_image`, and
  `submit_subagent_report`; they cannot call other agents or submit the task.
- Report schema remains small: `status`, `report`, and `unresolved`.
- Verifier must perform at least one inspection call before its report is accepted.

### Shared budget

- Actor, Binder, Critic, Verifier, and compaction consume one logical-call budget.
- Source invocation caps remain Binder 40, Critic 32, and Verifier 24 per call;
  the Actor may call Binder twice, Critic twice, and Verifier three times.
- Critic is rejected until a Binder report exists.
- Requested allocations are clamped by the role cap, profile maximum, remaining
  shared calls, and a one-call Actor reserve.
- Normal submission requires at least one report from all three roles. Exhaustion
  records missing roles and forces truthful completion.
- Do not add a Verifier reserve, mandatory repair, mandatory recheck, fixed phase
  allocations, or another role in the parity implementation.

### Compaction and tracing

- Compaction consumes the same shared budget and starts a new Actor epoch.
- Stable system instructions, original task, initial public images, and tool order
  are reconstructed exactly; only the compact handoff changes.
- Budget counters, role counts, reports, provenance, and activation history remain
  harness-owned state, not claims inside the LLM summary.
- Write `orchestration.json`, `orchestration-events.jsonl`, raw/sanitized model
  request artifacts, and `subagents/NN-role.json`.
- Redact credentials, cookies, API keys, and large image data from human-readable
  traces while retaining hashes and separately bounded blobs.

### Responses gateway scope

- V1 uses GPT-5.5 through the Responses-compatible PhyAGI/AIServices path with
  `reasoning_effort=xhigh`, strict sessions, unique session ids, actual function
  calls, and `parallel_tool_calls=false`.
- Extract a small `ResponsesToolModel`/transport interface from `PhyagiModel` so
  its credential, endpoint, infrastructure, retry, and usage-accounting logic is
  shared without passing output through the XML/JSON action parser.
- Leave Anthropic, OpenRouter chat-completions, vLLM, and SFT-state models on the
  existing single-agent path until separately tested for tool-loop compatibility.

</details>

<details>
<summary>3. Runtime boundary and proposed repository layout</summary>

Define one benchmark-neutral runtime protocol. Role and stage are harness metadata,
not model-supplied arguments.

```python
class CliTaskRuntime(Protocol):
    task_id: str
    instruction: str
    initial_image_paths: tuple[str, ...]
    provenance: dict[str, object]

    def run_command(
        self, command: str, *, timeout: int, role: str, stage: str
    ) -> CommandResult: ...

    def read_file(self, path: str, *, role: str, stage: str) -> bytes | None: ...
    def finalize(self) -> dict[str, object]: ...
    def close(self) -> None: ...
```

`CommandResult` normalizes `status`, `stdout/output`, `stderr/error`, return code,
timeout, truncation, infrastructure-failure classification, and artifact metadata.
The concrete runtime owns all post-command hooks, so CadGenBench checkpointing cannot
be accidentally skipped by an Actor or specialist call.

```text
src/miniswewebagent/
  agents/
    cli_actor_orchestrator.py
  orchestration/
    __init__.py
    budget.py
    prompts.py
    profiles.py
    reports.py
    responses_thread.py
    runtime.py
    runner.py
    trace.py
  benchmarks/
    __init__.py
    docker_exec.py
    image_validation.py
    cadgenbench/
      __init__.py
      cad_mcp_cli.py
      container/
      container_tools.py
      environment.py
      errors.py
      packaging.py
      prompts.py
      public_eval_job.py
      public_prompts.py
      task.py
    benchcad/
      __init__.py
      container/
      docker_exec.py
      environment.py
      errors.py
      image.py
      prompts.py
      scorer_cli.py
      scoring.py
      task.py
  config/generation/
    cli_actor_orchestrator_gpt55.yaml
  run/benchmarks/
    cadgenbench.py
    benchcad.py
  run/utilities/
    benchcad_score.py
tests/
  test_cli_orchestrator_*.py
  test_cadgenbench_*.py
  test_benchcad_*.py
```

### Adapter use

- `LocalWorkspaceRuntime` wraps the current `LocalWorkspaceEnvironment.execute()`
  and adds role/stage metadata without changing existing callers.
- `CadGenBenchEnvironment` and `BenchCADEnvironment` implement the runtime protocol
  directly; do not force their Docker lifecycle through browser-specific methods.
- Prompt selection, initial images, task limits, and tool descriptions live in a
  `BenchmarkProfile`. The main runner must not grow `if benchmark == ...` branches.
- Evaluation and cleanup are lifecycle hooks, never model tools.

### Packaging and dependencies

- Add the OpenAI Responses client and Docker SDK as explicit dependencies or a
  documented `cadbench` optional extra; do not rely on transitive imports.
- Keep `mcp>=1.9,<2`, Build123d, CadQuery, and benchmark packages inside their
  pinned container images rather than installing CAD stacks into mini-web's host
  environment.
- Include Dockerfiles, wrapper scripts, prompts, and other container assets in
  package data and in source distributions.
- Add Apache-2.0 license/notice attribution for copied DeskWright code and retain
  comparator prompt provenance at its pinned commit.

</details>

<details>
<summary>4. CadGenBench adapter track</summary>

### Selective source port

Port and re-namespace the branch's task, environment, Docker exec, image validation,
MCP CLI, container tools, public prompts/evaluator, packaging, error types, Dockerfile,
build script, manifest preparation, and submission packaging. Port only the
Actor-orchestrator prompt slice:

- CAD Actor guidance;
- CAD orchestrator supplement;
- fresh Binder, Critic, and Verifier prompts;
- CAD compaction handoff.

Do not port fixed bindings, Binder adjudication, fixed audit, repair extraction, or
recheck prompt machinery even though older definitions coexist in the branch.

### Pinned task/runtime contract

- CadGenBench source commit:
  `33304cf771fc5639144b1df9611e347251052cf8`.
- Dataset revision:
  `f76f965585817c621d6ea0d150d745adf670e66e`.
- Container tag/label: `deskwright-cadgenbench:33304cf` and
  `org.cadgenbench.commit`.
- Build123d-MCP 0.3.59, Build123d 0.11.0, drafting helpers 0.12.0,
  `bd_warehouse` 0.2.0, and Augura 0.1.5.
- Shared hard maximum/default: 150 Responses calls; compact every 40.
- Build123d execute timeout 120 seconds, outer MCP command timeout 600 seconds,
  and six-hour emergency infrastructure cutoff.

### Model-facing CAD access

- Start one persistent Build123d-MCP streamable-HTTP service per fixture at
  `127.0.0.1:8765` inside the task container with no published Docker port.
- The Responses API never receives MCP schemas. Actor and specialists use only:

```text
cad-mcp tools
cad-mcp schema TOOL
cad-mcp call TOOL --json JSON
cad-mcp execute /workspace/source.py
```

- Keep generated MCP images under `/workspace/.cad-mcp/images` and attach them
  only through `read_image`.
- Preserve the source's prompt-level specialist non-mutation boundary. A stronger
  technical read-only specialist sandbox would be a separate experiment.

### Inputs and isolation

- Generation: stage and attach every declared `input*.png`, not just `input.png`.
- Editing: stage `input.step`, `edit_description.txt`, mesh sidecar, and public
  front/top/right/iso renders; attach all four renders.
- Attach the immutable public image set to the main Actor and every fresh specialist,
  including after compaction.
- Validate safe fixture ids, declared paths, file sizes, SHA-256 values, clean pinned
  benchmark checkout, and dataset metadata revision before model calls.
- Run with no network, read-only root and `/input`, non-root UID/GID, all capabilities
  dropped, `no-new-privileges`, 12 GiB RAM, four CPUs, 1,024 PIDs, bounded tmpfs,
  and no credentials, ground truth, evaluator internals, scores, prior submissions,
  or upload route.

### Last-valid checkpoint and recovery

- After **every** shell command, detect changes to canonical `output.step`/`output.stp`,
  public-validate the candidate, and archive it only if valid.
- Before evaluation, atomically restore the latest protected valid STEP. Failed
  exports, policy rejection, model exhaustion, MCP restart, container recreation,
  and emergency cutoff must not destroy that checkpoint.
- Preserve the bounded MCP restart limit, stopped-container recreation, persistent
  volume restoration, lifecycle monitoring, peak resource telemetry, failure
  snapshots, and cleanup-on-startup-failure behavior from the branch.

### Finalization

- Freeze bounded artifacts, restore the last valid candidate, then run only public
  validity and multi-view rendering in trusted host-side processing.
- Return truthful statuses: `pending_official_score`, `invalid_output`,
  `missing_output`, or `load_failed`; never fabricate a private benchmark score.
- Port the provenance-checked submission packager producing `meta.json` plus one
  `<fixture>/output.step` per fixture. Upload remains a separate manual action.

</details>

<details>
<summary>5. BenchCAD adapter track</summary>

### What is and is not ported

Port the branch's Vision2Code task/manifest layer, hardened task environment, bounded
Docker exec, image-label validation, container image and CLI wrappers, freeze/copy
logic, scorer CLI, two-container scoring path, manifest preparation, and offline score
entry point.

Do **not** port:

- `BudgetPolicy` or any fixed role allocation;
- `Binder -> Critic -> Binder adjudication -> Actor -> audit -> repair -> recheck`;
- fixed binding/patch/verification schemas;
- `run_benchcad_harness.py --mode full` semantics;
- `direct.py` or a `--mode direct` agent path in this integration.

BenchCAD gets a new `BenchCADProfile` for the common Actor-orchestrator. Its prompts
combine the generic dynamic role contract with only the benchmark-specific public
instruction, CadQuery CLI, output, validation, and isolation rules. The old fixed-flow
prompts are reference material, not code to reuse verbatim.

### Pinned task/runtime contract

- BenchCAD commit:
  `52087ef4b08811c21c09b8b16d0ae36c70844865`.
- Smoke dataset revision:
  `5919f578ab09ec283603a082fab07c7639ab56eb`.
- Container tag/label: `deskwright-benchcad:52087ef` and
  `org.benchcad.commit`.
- Initial profile: GPT-5.5 xhigh, shared hard maximum/default 300 Responses calls,
  compact every 40. This matches the source compute ceiling but not its topology.
- Only `/input/input.png` is model-visible. Ground truth paths and hashes remain
  host-only provenance.

### Actor CLI

The Actor writes `/workspace/prediction.py`, binding final CadQuery geometry to
`result`, and iterates through ordinary shell commands:

```text
benchcad-run /workspace/prediction.py --step /workspace/candidate.step
benchcad-render /workspace/candidate.step /workspace/candidate.png
```

Neither command exposes target geometry, evaluator source, or score. The initial input
image is attached to the Actor and every fresh specialist; candidate renders are read
through the same bounded `read_image` tool.

### Isolation and final artifact capture

- Agent container: no network, no credentials/evaluator/ground truth/dataset cache,
  read-only root and `/input`, private size-limited 2 GiB tmpfs Docker volume at
  `/workspace`, non-root mapped UID/GID, no capabilities, and
  `no-new-privileges`.
- Capture at most 1 MiB per stdout/stderr stream and bound every file read.
- On finalization, pause the agent container and use a trusted sidecar to copy only
  bounded regular `prediction.py`, `candidate.step`, and `candidate.png` artifacts.
- Remove the task container and volume before scoring, even when snapshotting fails.

### Trusted scoring boundary

Use two fresh, networkless containers:

1. Execute the immutable submitted `prediction.py` without ground truth or scorer
   source and copy out the generated STEP.
2. Score that STEP against the immutable ground truth without mounting submitted
   Python code.

Retain the exact prediction SHA-256, generated STEP, raw prediction/execution record,
ground-truth SHA-256, immutable image id, benchmark commit, source snapshot, and raw
scorer result. Preserve truthful zero/status handling for missing code and model-policy
failures; retry only classified scoring infrastructure failures.

`mini-web-benchcad score` runs this trusted path for a deferred run directory. It is
post-processing and cannot feed results back into the Actor. Immediate scoring may be
supported by omitting `--defer-score`, but generation and scoring telemetry remain
separate.

</details>

<details>
<summary>6. CLI, configuration, and artifact behavior</summary>

### Selection rules

- Add `--actor-orchestrator`, `--total-steps`, and `--compact-every` to `mini-web`
  and `mini-web-om2w`. Without the flag, the resolved current `agent_class` and all
  single-agent behavior remain unchanged.
- Add Typer groups `mini-web-cadgenbench` and `mini-web-benchcad`. Their `run`
  commands instantiate `CliActorOrchestratorAgent` unconditionally and expose no
  agent-mode selector.
- CLI values override YAML. Benchmark hard ceilings cannot be raised by YAML or CLI.
- Use one `generation/cli_actor_orchestrator_gpt55.yaml` for the model and common
  agent contract; keep benchmark paths, image pins, and task ids in CLI/manifests.

### Supporting commands

- `mini-web-cadgenbench prepare-manifest` validates and freezes fixture provenance.
- `mini-web-cadgenbench package` creates an official-layout submission archive but
  never uploads it.
- `mini-web-benchcad prepare-manifest` freezes record selection and hashes.
- `mini-web-benchcad score` scores only immutable deferred artifacts.
- Container build helpers refuse dirty or wrong-commit source checkouts and validate
  image labels before any model call.

### Artifact boundaries

- For mini-web/OM2W, only main Actor `run_command` calls enter benchmark action
  history. Specialist inspection, role tools, compaction, and control events are
  orchestration telemetry.
- CadGenBench and BenchCAD keep command provenance for every role because environment
  safety hooks run after every command, while clearly labelling main Actor versus
  specialist calls.
- Existing `trajectory.json`, `result.json`, screenshots, and OM2W judge exports stay
  backward compatible.
- CAD run directories additionally store task/manifest provenance, immutable hashes,
  container identity, checkpoint/scorer records, resource telemetry, and finalization
  status.

### Failure semantics

- Retry Responses calls only for typed transient gateway failures and never double
  count a successfully completed logical call.
- A malformed tool argument returns a tool error; it cannot execute a partial command.
- Distinguish model-policy rejection, task invalidity, missing output, command failure,
  timeout, container/MCP infrastructure failure, and scoring infrastructure failure.
- Retry an entire task only for a classified infrastructure failure, under a new run
  directory linked to the failed attempt. Never overwrite or silently select attempts.
- Cleanup runs in `finally` paths for browser sessions, Docker containers, volumes,
  MCP services, and scorer containers.

</details>

<details>
<summary>7. Implementation phases and exit gates</summary>

### Phase A — Freeze source slices and add parity fixtures

- [ ] Record source branch tips, actor behavior commit/tag, benchmark commits, dataset
  revisions, image labels, prompt sources, and copied-file attribution.
- [ ] Build a file-by-file port inventory; explicitly mark fixed-runner and direct-mode
  files excluded.
- [ ] Copy representative source test fixtures before changing imports or interfaces.
- [ ] Add golden tests for the six Actor tools, three report schemas, budget snapshots,
  and benchmark provenance manifests.

Exit gate: every copied component has an exact source ref and an owning mini-web module.

### Phase B — Common Responses Actor-orchestrator

- [ ] Add Responses function-call normalization, tool outputs, stable prefix handling,
  strict sessions, retries, token usage, and trace writing.
- [ ] Implement `SharedStepBudget`, dynamic Actor dispatch, generic fresh specialist
  runner, invocation constraints, forced final calls, and compaction.
- [ ] Add `CliTaskRuntime`, `BenchmarkProfile`, normalized command results, and a
  `LocalWorkspaceRuntime` adapter.
- [ ] Wire the explicit flag into `mini-web` and `mini-web-om2w` without changing the
  no-flag path.

Exit gate: fake-gateway tests prove the runner never calls a specialist unless the
Actor selected that tool, and no fixed-stage method exists.

### Phase C — CadGenBench environment

- [ ] Port/re-namespace task discovery, manifest verification, public prompts,
  container assets, MCP CLI, environment, image validation, and errors.
- [ ] Implement all-public-image attachment to every fresh context.
- [ ] Preserve last-valid checkpointing, MCP recovery, stopped-container recreation,
  emergency cutoff, resource telemetry, and cleanup.
- [ ] Add public finalization and submission packaging.
- [ ] Add the dedicated CLI group with the 150-call hard ceiling.

Exit gate: one generation fixture and one editing fixture complete through the common
Actor runner, restore a valid checkpoint, and produce truthful public-evaluation output.

### Phase D — BenchCAD environment and scorer

- [ ] Port/re-namespace manifest/task validation, container assets, environment,
  bounded exec/read, image validation, freeze/copy, scorer CLI, and scorer isolation.
- [ ] Add the new dynamic `BenchCADProfile`; do not import fixed role prompts or runner.
- [ ] Attach only the public input image to each fresh context.
- [ ] Add `run`, `prepare-manifest`, and trusted offline `score` commands.
- [ ] Add the 300-call profile ceiling and deferred-score artifact contract.

Exit gate: a smoke record produces an immutable `prediction.py`, executes in the first
trusted container, and scores in a separate ground-truth container with no cross-mount.

### Phase E — Reliability, packaging, and documentation

- [ ] Declare optional dependencies, package container assets, update the lockfile,
  and add license/notice attribution.
- [ ] Add no-model image/MCP/container preflights and cleanup fault injection.
- [ ] Document exact commands, cache layout, disk/memory prerequisites, provenance,
  score boundaries, retry policy, and artifact schemas.
- [ ] Verify installed-wheel execution, not just a source checkout.

Exit gate: every command works from a clean install and failure injection leaves no
owned browser session, container, volume, or scorer process behind.

</details>

<details>
<summary>8. Test matrix and verification commands</summary>

### Common unit/integration tests

- shared budget and per-role accounting never exceed the profile ceiling;
- allocation clamping, invocation limits, Binder-before-Critic, required roles, and
  forced reports/submission match the pinned Actor-orchestrator;
- stable tool order and original task/images survive compaction;
- role contexts and Actor epochs use unique strict-session ids;
- Verifier cannot submit before inspecting;
- malformed/truncated function JSON cannot execute;
- gateway retries do not duplicate logical calls or task commands;
- specialist calls never enter OM2W action history;
- no-flag mini-web/OM2W configuration and behavior remain unchanged;
- dedicated CAD CLIs expose no sequential/direct agent mode.

### Fake-gateway scenarios for every benchmark profile

1. Actor command → Binder → command → Critic → command → Verifier → submit.
2. Actor repairs after Verifier and submits without a second Verifier.
3. Actor chooses a second Verifier after repair.
4. Actor attempts Critic before Binder and receives a tool error.
5. Actor attempts early submission and receives missing-role feedback.
6. Budget exhausts inside a specialist and yields a forced truthful report.
7. Compaction occurs between specialist calls and preserves live runtime state.
8. Environment/tool failure releases every owned resource.

### CadGenBench-specific coverage

- generation and editing discovery, unsafe-path rejection, revision/hash tamper checks;
- exact command-only MCP schema and every declared image attached on first turns;
- hardened container configuration and image-label validation;
- valid candidate archived, invalid candidate rejected, last-valid candidate restored;
- MCP restart and container recreation restore the protected workspace;
- deadline/model-policy paths still finalize the last valid candidate;
- missing/invalid/load-failed statuses and provenance are truthful;
- submission packager rejects missing fixtures, duplicates, or provenance mismatches.

Selectively port the branch's targeted CadGenBench safety tests, including lifecycle,
checkpoint, recovery, packaging, and public-input cases, rather than only writing new
happy-path tests.

### BenchCAD-specific coverage

- safe record ids, manifest asset/hash validation, and exact image label;
- only input image mounted into the task container; ground truth never is;
- bounded output/read behavior and tmpfs archive fallback;
- pause/freeze/sidecar copy and cleanup on snapshot failure;
- missing prediction returns truthful zero without starting unnecessary scorer work;
- submitted-code execution container cannot see ground truth/scorer source;
- score container cannot see submitted Python;
- exact hashes and raw scorer artifacts survive deferred/offline scoring;
- only classified scorer infrastructure failures retry.

### Local verification

```bash
pytest -q tests/test_cli_orchestrator_*.py
pytest -q tests/test_cadgenbench_*.py
pytest -q tests/test_benchcad_*.py
pytest -q
ruff check src tests
python -m build
```

### No-model runtime preflight

```bash
mini-web-cadgenbench preflight \
  --dataset-dir cache/cadgenbench-data \
  --cadgenbench-repo cache/cadgenbench \
  --image deskwright-cadgenbench:33304cf

mini-web-benchcad preflight \
  --manifest cache/benchcad-smoke/manifest.json \
  --benchcad-repo cache/BenchCAD-main \
  --image deskwright-benchcad:52087ef
```

The CadGenBench preflight must verify the streamable-HTTP transport, MCP version and
complete tool list at intended concurrency. Both preflights must prove mounts, labels,
network isolation, resource limits, bounded writes, and cleanup without consuming a
model call.

</details>

<details>
<summary>9. Evaluation and promotion gates</summary>

Follow the experiment rules in the referenced `AGENTS.md`: freeze inputs and config,
preserve all attempts, retry only infrastructure failures, and inspect task-level
causes rather than promoting on aggregate movement.

### CadGenBench

1. Run no-model preflight at width 1, then the intended smoke width.
2. Run one generation and one editing fixture at width 1.
3. Run the pinned eight-fixture panel `101,113,127,138,150,203,208,242` at width
   no greater than 8 for the first port validation.
4. Compare source-versus-port prompt hashes, tool schemas, role activations, calls,
   checkpoint validity, statuses, tokens, cost, and wall time.
5. Expand to all 81 tasks only after every output is valid and lifecycle telemetry
   shows no unexplained MCP/container failure.

Do not claim the source 0.4979 score until the port has an independently evaluated,
fully pinned run.

### BenchCAD

1. Run one frozen smoke record at width 1 with deferred scoring.
2. Run the branch's frozen five-record smoke selection and score it offline.
3. Load-test generation and scoring separately at width 8 before raising concurrency.
4. Freeze one Actor-orchestrator configuration and run a predeclared panel; do not tune
   on those evaluation records.
5. A 212-record run may compare operationally with the banked branch artifact, but
   report the topology difference and do not call it a reproduction or compute-matched
   comparison.

### Required run report

Record commit, dirty state, model, reasoning effort, gateway infrastructure, task ids,
manifest/dataset revisions, image ids, total calls, compactions, calls/tokens by role,
cached tokens, retries, wall time, peak resources, checkpoints, output validity,
prediction execution status, IoU where available, attempt lineage, and leaked-resource
audit. Inspect every regression against a prior valid/source result.

Do not change `mini.yaml`, default OM2W configs, benchmark pins, role prompts, budget
rules, or scoring behavior during a frozen panel. Any new reserve, role, recheck gate,
technical specialist sandbox, or scoring feedback mechanism is a separate experiment.

</details>

<details>
<summary>10. Explicit exclusions and definition of done</summary>

### Excluded

- Fixed Binder/Critic/adjudication/Actor/audit/repair/recheck scheduling.
- `BudgetPolicy`, `PatchState`, structured startup bindings, repair-item extraction,
  automatic post-Actor audit, and mandatory recheck.
- BenchCAD `direct.py` and any agent-mode `--mode full|direct` switch.
- Direct Responses exposure of Build123d-MCP or BenchCAD CAD operations.
- Host-side MCP stdio bridge or a Docker-published MCP port.
- Ground truth, evaluator source, score feedback, prior submissions, or hidden benchmark
  artifacts in a model-visible container/context.
- OSWorld `DesktopEnv`, QCOW2/provider/setup/evaluator infrastructure, multi-phase
  desktop handling, Claude client, Planner, self-ledger, or new specialist roles.
- Automatic upload of a CadGenBench submission.

### Definition of done

- [ ] One shared persistent Actor runner serves web, CadGenBench, and BenchCAD.
- [ ] The Actor alone chooses every specialist call and allocation under one budget.
- [ ] The only main model tool surface is the stable six-tool CLI contract.
- [ ] No fixed/sequential multi-agent implementation is added to mini-web.
- [ ] Dedicated CAD commands always choose Actor-orchestrator and offer no fallback.
- [ ] CadGenBench uses fixture-local `cad-mcp` CLI, all public images, protected
  last-valid checkpoints, truthful public evaluation, and provenance-safe packaging.
- [ ] BenchCAD uses only public input during generation and preserves the hardened
  freeze plus two-container scoring separation.
- [ ] Existing mini-web/OM2W single-agent paths remain backward compatible when the
  explicit flag is absent.
- [ ] Full unit, fake-gateway, Docker preflight, failure-injection, package-build, and
  clean-install tests pass.
- [ ] Every owned browser/Docker/MCP/scorer resource is released on all exit paths.
- [ ] Documentation clearly separates source results from new port results and records
  all licenses, pins, and claim boundaries.

</details>
