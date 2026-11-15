# CUAWright CadGenBench Arm B Pilot

## Scope

This integration now implements only the corrected **Arm B** pilot:

```text
Main Actor-orchestrator
  -> read-only Binder
  -> Actor execution
  -> independent read-only Verifier gate
  -> one final candidate
```

The main Actor owns every CAD mutation through one persistent
**Build123d-MCP 0.3.59** session inside the isolated fixture container.
Every model role receives only `run_command`. Image attachment, role calls,
reports, and submission are controlled through terminal `agent` commands and
files. CAD operations go through the fixture-local `cad-mcp` command. Binder and Verifier
may inspect with that command but are prompted not to mutate judge-relevant
state.

Pinned public references:

- CadGenBench source commit:
  `33304cf771fc5639144b1df9611e347251052cf8`
- Public input snapshot revision:
  `f76f965585817c621d6ea0d150d745adf670e66e`
- Comparator prompt source:
  `pzfreo/cadgenbench-build123d` commit
  `0a8a78da20c726181a6210924096a8c208eafd4a`
- Frozen eight-fixture panel:
  `101,113,127,138,150,203,208,242`

## Benchmark rules vs pilot choices

Benchmark/public contract:

- one fixture attempt yields one final `output.step`;
- public validity/render checks are trusted host-side post-processing;
- missing or invalid outputs stay truthful and score zero.

Empirical pilot choices for this run package:

- model: `gpt-5.5`, reasoning effort `xhigh`;
- shared hard cap: **150 model responses** across Actor, Binder, Critic,
  Verifier, and compaction;
- provider token usage is recorded and summarized, but **never** used as a
  stopping rule;
- Build123d `execute()` timeout: `120s`;
- outer MCP tool timeout: `600s`;
- no authoring/scoring wall-clock cutoff beyond a **6 hour emergency
  infrastructure stop**;
- width is external to the per-fixture process; the recommended launcher below
  uses width `8`.

The `150` response cap is a CUAWright pilot choice derived from the public
comparator’s observed maximum of `131` assistant messages. It is **not** a
published CadGenBench rule.

## Build the hardened image

The fixture container keeps the official CadGenBench package and additionally
installs `build123d-mcp==0.3.59` into the same Python 3.12 environment.

```bash
bash scripts/tools/build_cadgenbench_image.sh \
  cache/cadgenbench cuawright-cadgenbench:33304cf
```

The build still refuses a dirty or wrong-commit CadGenBench checkout. Runtime
provenance records the CadGenBench commit, the Build123d-MCP version, container
image ID, and MCP health/version startup checks.

## Isolation and transport

Each fixture gets exactly one live task container and exactly one persistent
Build123d-MCP session:

- `network_mode=none`;
- read-only root filesystem;
- read-only `/input`;
- non-root UID/GID;
- `cap_drop=["ALL"]`;
- `no-new-privileges`;
- bounded CPU, memory, PID count, and tmpfs (`12 GiB` / `1024` PIDs for the
  task container as an isolation resource bound, not model compute);
- no credentials, ground truth, evaluator internals, or upload path;
- Build123d-MCP runs as the container service over streamable HTTP bound only
  to `127.0.0.1:8765`; no Docker port is published.

The host verifies service health, version, and tool discovery before model
work. The Responses API never receives MCP tool schemas. Actor and specialists
use:

```text
cad-mcp tools
cad-mcp schema TOOL
cad-mcp call TOOL --json JSON
cad-mcp execute /workspace/source.py
```

MCP images are written under `/workspace/.cad-mcp/images` and inspected with
`python /opt/cuawright-tools/image_show.py --path PATH` through `run_command`.

Provenance explicitly records that the model-visible environment excludes:

- public submission outputs/candidates from prior systems;
- public scores or official per-fixture score breakdowns;
- public trajectory logs such as `gpt55-v0359-xhigh-full`.

Those artifacts may be compared later by a host-side analyst, but they are not
mounted or shown to the model during generation.

## Prompt and input alignment

The main Actor receives the exact public generic generation/editing prompts from
the comparator repository as task-level CAD guidance. CUAWright adds only a
short orchestration note: the Actor owns mutations, specialist reports are
advisory, and every model call shares the same budget.

Public input policy uses every fixture-declared public drawing:

- generation: stage and attach `input.png`, `input2.png`, and any other
  declared generation drawings;
- editing: stage `input.step`, `edit_description.txt`, and attach
  `renders/front.png`, `renders/top.png`, `renders/right.png`, `renders/iso.png`.

The main Actor and every fresh specialist see those initial images in their
first prompt context. They may also inspect public files and the live CAD state
with `run_command`, the provisioned `image_show.py` client, and `cad-mcp`.

## Last-valid checkpoint semantics

After every `run_command`, the host detects a changed canonical
`output.step`/`output.stp`, independently validates it, and archives it only
when valid. Later failed `execute`, failed export, budget exhaustion, policy
rejection, or emergency cutoff cannot delete that checkpoint.

Before trusted public evaluation, the harness restores the archived latest-valid
STEP to canonical `workspace/output.step`. If no valid export ever existed, the
run truthfully reports `missing_output`.

## Run one fixture

```bash
uv run cuawright-cadgenbench \
  --dataset-dir cache/cadgenbench-data \
  --fixture-id 101 \
  --image cuawright-cadgenbench:33304cf \
  --cadgenbench-repo cache/cadgenbench \
  --api-key-file /data/iso-osworld-v2/.a-creds \
  --result-root results-cadgenbench \
  --model gpt-5.5 --reasoning-effort xhigh \
  --total-steps 150
```

The CadGenBench runner now defaults to Actor-orchestrator mode and does **not**
silently fall back to the staged Binder/Critic/Actor flow.

## Width-8 launcher

```bash
root="results-cadgenbench-panel8-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$root"
printf '%s\n' 101 113 127 138 150 203 208 242 > "$root/fixture_ids.txt"

run_one() {
  fixture="$1"
  uv run cuawright-cadgenbench \
    --dataset-dir cache/cadgenbench-data \
    --fixture-id "$fixture" \
    --image cuawright-cadgenbench:33304cf \
    --cadgenbench-repo cache/cadgenbench \
    --api-key-file /data/iso-osworld-v2/.a-creds \
    --result-root "$root" \
    --model gpt-5.5 --reasoning-effort xhigh \
    --total-steps 150 \
    >"$root/process-$fixture.log" 2>&1
}
export -f run_one
xargs -r -n1 -P8 bash -c 'run_one "$1"' _ < "$root/fixture_ids.txt"
```

## Telemetry

Each run records:

- configured limits and retry policy;
- actual model responses by role and stage;
- provider input/cached/output/reasoning tokens;
- MCP HTTP startup, version, health, and available-tool checks;
- checkpoint history and restored latest-valid candidate hash;
- per-task `container-telemetry.jsonl`, lifecycle snapshots, failure summaries,
  peak memory / PIDs, exit states, and recreation count;
- wall-clock timestamps;
- container image and source provenance.

## Local evaluation semantics

Trusted host-side post-processing remains public only:

1. freeze bounded artifacts from the mutable workspace;
2. restore the archived latest-valid `output.step`, if one exists;
3. run the public CadGenBench validity evaluator;
4. render fresh public multi-view candidate images.

Returned statuses stay truthful:

- `pending_official_score`
- `invalid_output`
- `missing_output`
- `load_failed`

No local run fabricates private leaderboard metrics.

## Package a submission zip

```bash
python3 scripts/tools/package_cadgenbench_submission.py \
  --manifest cache/cadgenbench-panels/panel-8.json \
  --run-dir results-cadgenbench/task-101-... \
  --run-dir results-cadgenbench/task-113-... \
  --run-dir results-cadgenbench/task-127-... \
  --run-dir results-cadgenbench/task-138-... \
  --run-dir results-cadgenbench/task-150-... \
  --run-dir results-cadgenbench/task-203-... \
  --run-dir results-cadgenbench/task-208-... \
  --run-dir results-cadgenbench/task-242-... \
  --output results-cadgenbench/panel-8-submission.zip \
  --submitter "Your Name" \
  --name "CUAWright CadGenBench Arm B panel-8"
```

`agree_to_publish` remains `false` unless `--agree` is passed. Upload is still a
separate manual step outside this harness.
