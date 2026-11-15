# CUAWright BenchCAD Vision2Code Adapter

The banked GPT-5.5 xhigh 212-record result is documented in
[`CLI_HARNESS_BENCHCAD_212_RESULT.md`](CLI_HARNESS_BENCHCAD_212_RESULT.md).

## Scope

The canonical unified runtime uses the persistent Actor orchestrator:

```text
Main Actor -> fresh Binder -> Actor execution -> independent fresh Verifier
-> Actor repair/recheck -> terminal submission -> scorer
```

The only model-facing tool is `run_command`. Specialist calls, image
attachments, reports, and submission are terminal-controlled `agent` commands
and files. BenchCAD candidate execution and rendering remain ordinary
`benchcad-run` and `benchcad-render` commands.

The banked 212-record result predates this unified transport and retains its
historical fixed-pipeline provenance.

This is an agentic tool condition, not BenchCAD's canonical direct/no-tool
condition. Report the official voxel IoU metric and all resource usage; do not
claim compute-matched comparison with provider runs whose tool budgets are not
published.

## Pinned runtime

- BenchCAD commit:
  `52087ef4b08811c21c09b8b16d0ae36c70844865`
- Smoke dataset revision:
  `5919f578ab09ec283603a082fab07c7639ab56eb`
- Image tag: `cuawright-benchcad:52087ef`
- Image commit label: `org.benchcad.commit`

The image label is checked before model calls and scoring. The scorer archives
the exact Git commit rather than importing from a mutable checkout.

Build with Docker data and caches on `/data`:

```bash
export UV_CACHE_DIR=/data/iso-osworld-v2/.uv-cache
export HF_HOME=/data/iso-osworld-v2/.hf-cache
bash scripts/tools/build_benchcad_image.sh \
  cache/BenchCAD-main cuawright-benchcad:52087ef
```

## Isolation

The agent container has:

- no network;
- no credentials, evaluator, ground truth, or dataset cache;
- a read-only root filesystem and `/input`;
- a private 2 GiB size-limited tmpfs Docker volume at `/workspace`;
- a non-root host-mapped UID/GID and no Linux capabilities.

Command output is streamed and captured up to 1 MB per stream. File reads are
bounded. At evaluation, the agent container is paused, a trusted sidecar copies
only bounded regular final artifacts, and the agent container and workspace
volume are removed.

Final scoring uses two separate containers:

1. submitted code executes without ground truth or scorer source;
2. the generated STEP is scored against ground truth without submitted code.

The SHA-256 of the exact scored `prediction.py`, immutable image ID, benchmark
commit, raw prediction, generated STEP, and scorer result are retained.

## Candidate CLI

Inside the agent container:

```bash
benchcad-run /workspace/prediction.py \
  --step /workspace/candidate.step

benchcad-render /workspace/candidate.step \
  /workspace/candidate.png
```

Neither command exposes target geometry or official score feedback.

## Prepare a manifest

The manifest contains host-only paths to public inputs and ground truth. Only
`/input/input.png` is mounted into the agent container.

```bash
uv run scripts/tools/prepare_benchcad_manifest.py \
  --records-jsonl cache/BenchCAD-main/Vision2Code/test_data/records.jsonl \
  --data-dir cache/BenchCAD-main/Vision2Code/test_data \
  --output cache/benchcad-smoke/manifest.json \
  --benchcad-repo cache/BenchCAD-main \
  --image cuawright-benchcad:52087ef
```

The BenchCAD checkout must be clean. For scored panels, pin and record the HF
dataset revision and publish the deterministic record-selection manifest before
model execution.

## Run

Full CUAWright:

```bash
uv run cuawright-benchcad \
  --manifest cache/benchcad-smoke/manifest.json \
  --record-id washer_000001_s20260505 \
  --mode full \
  --image cuawright-benchcad:52087ef \
  --benchcad-repo cache/BenchCAD-main \
  --api-key-file /data/iso-osworld-v2/.a-creds \
  --result-root results-benchcad \
  --model gpt-5.5 --reasoning-effort xhigh \
  --total-steps 300 \
  --defer-score
```

Canonical direct control:

```bash
uv run cuawright-benchcad \
  --manifest cache/benchcad-smoke/manifest.json \
  --record-id washer_000001_s20260505 \
  --mode direct \
  --image cuawright-benchcad:52087ef \
  --benchcad-repo cache/BenchCAD-main \
  --api-key-file /data/iso-osworld-v2/.a-creds \
  --result-root results-benchcad-direct \
  --model gpt-5.5 --reasoning-effort xhigh \
  --defer-score
```

Score the immutable prediction later at the scorer panel's independent width:

```bash
uv run scripts/python/score_benchcad_prediction.py \
  --manifest cache/benchcad-smoke/manifest.json \
  --record-id washer_000001_s20260505 \
  --run-dir results-benchcad/task-washer_000001_s20260505-... \
  --image cuawright-benchcad:52087ef \
  --benchcad-repo cache/BenchCAD-main
```

## Scored-panel plan

1. Validate fixture tasks at width 1.
2. Load-test widths 8 and 23 without consuming new evaluation IDs.
3. Freeze width 23 if infrastructure remains reliable; otherwise freeze 16
   before scored work starts.
4. Run one canonical-direct and one Full CUAWright pass with `--defer-score`
   on the same frozen 1,000-record family-by-difficulty manifest.
5. Run `score_benchcad_prediction.py` offline at width 8, separate from agent
   execution.
6. Extend to all 17,900 records only after cost and reliability review.

Use unique result roots, preserve every attempt, and retry only classified
infrastructure failures.

## Five-record smoke result

The smoke used the first five records from the first pinned dataset shard. It is
an integration check, not a representative benchmark result.

| Record | Direct IoU | Full IoU |
| --- | ---: | ---: |
| `washer_000001_s20260505` | 0.294118 | 1.000000 |
| `hex_nut_000006_s20260505` | 0.983364 | 1.000000 |
| `spacer_ring_000007_s20260505` | 0.509075 | 0.993035 |
| `bolt_000008_s20260505` | 0.632581 | 0.998408 |
| `washer_000005_s20260505` | 0.680699 | 0.998686 |
| **Mean** | **0.619967** | **0.998026** |

All ten final programs executed successfully under the hardened scorer.
Authoritative smoke scores are in each run's
`evaluation/rescore-hardened-v3/result.json`, summarized by
`cache/benchcad-smoke/final-smoke-report.json`.

Four Full runs completed while the scorer interface was being hardened and
their original immutable result files therefore record `score_fail`. Those
historical artifacts are intentionally preserved. The same final predictions
were re-executed and scored through the validated two-container path; direct
scores were unchanged and all hardened rescoring executions succeeded.
