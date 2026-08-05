# Script command guide

Use the concern-based paths below for commands and automation. Executable
scripts live in concern-specific subdirectories; the `scripts/` root contains
only this guide.

## Folders

- `archive/`: Old scripts kept for reference; do not use for active jobs.
- `cluster/`: Submit and run cluster evaluation jobs.
- `data/`: Prepare and tokenize training data.
- `eval/`: Evaluate completed agent runs.
- `lib/`: Shared helpers used by other scripts.
- `local/`: Serve models and run evaluations locally.
- `review/`: Serve results and open review tunnels.

## Command matrix

| Concern | Canonical command | Purpose |
| --- | --- | --- |
| Local OM2W | `bash scripts/local/om2w/serve_qwen35.sh` | Serve a local Qwen3.5 checkpoint with vLLM. |
| Local OM2W | `bash scripts/local/om2w/run.sh` | Run OM2W against one OpenAI-compatible local endpoint. |
| Local OM2W | `bash scripts/local/om2w/shards.sh <start\|stop\|status\|logs\|sessions>` | Split and manage a run across local vLLM ports. |
| Qwen3.5 cluster eval | `bash scripts/cluster/om2w/qwen35_9b/submit.sh [--dry-run]` | Submit the Qwen3.5-9B checkpoint evaluation. |
| Qwen3.5 cluster runtime | `bash scripts/cluster/om2w/qwen35_9b/run.sh` | Internal entry point used inside the Qwen3.5 job. |
| Qwen3.6 cluster eval | `bash scripts/cluster/om2w/qwen36_27b/submit.sh [--dry-run]` | Submit the Qwen3.6-27B evaluation. |
| Qwen3.6 cluster runtime | `bash scripts/cluster/om2w/qwen36_27b/run.sh` | Internal entry point used inside the Qwen3.6 job. |
| Judge-only cluster eval | `bash scripts/cluster/om2w/judge_only/submit.sh [--dry-run]` | Submit judging for existing run artifacts. |
| Judge-only cluster runtime | `bash scripts/cluster/om2w/judge_only/run.sh` | Internal entry point used inside the judge-only job. |
| Persistent evaluation | `python scripts/eval/persistent_cli.py --help` | Evaluate `browser-steps.jsonl` trajectories. |
| Persistent step evaluation | `python scripts/eval/persistent_cli_steps.py --help` | Evaluate `steps/*.sh` or `result.json` action histories. |
| Qwen3.5 data | `python scripts/data/qwen35/preprocess_lastobs_singleturn.py --help` | Build a last-observation, single-turn ShareGPT bundle. |
| Qwen3.5 data | `python scripts/data/qwen35/tokenize_lastobs_singleturn.py --help` | Tokenize the bundle for PhiTrain VLM SFT. |
| Review | `bash scripts/review/viewer.sh [port]` | Serve run and judge artifacts locally. |
| Review | `bash scripts/review/tunnel.sh [port]` | Create a public Cloudflare tunnel to the viewer. |

## Common examples

Serve a checkpoint and run one local evaluation:

```bash
CKPT=/path/to/checkpoint GPUS=0 TP=1 PORT=8000 \
  bash scripts/local/om2w/serve_qwen35.sh

CFG=eval/om2w_spb_vllm_sw10.yaml \
ENDPOINT=http://127.0.0.1:8000/v1 \
OUT=outputs/qwen35_om2w \
  bash scripts/local/om2w/run.sh
```

Manage a run sharded over two already-running servers:

```bash
PORTS="8000 8001" NTASKS=40 TASK_LEVEL=medium WORKERS=8 \
  bash scripts/local/om2w/shards.sh start
bash scripts/local/om2w/shards.sh status
bash scripts/local/om2w/shards.sh stop
```

Preview a cluster submission before creating a job:

```bash
bash scripts/cluster/om2w/qwen35_9b/submit.sh --dry-run
```

Serve and share the review viewer:

```bash
bash scripts/review/viewer.sh 8010
# In a second terminal:
bash scripts/review/tunnel.sh 8010
```

## Cluster SFT / eval tooling on branch `rl_yifei_clean`

The scripts in this section live on the `rl_yifei_clean` branch (repo-root
`docker/` and `scripts/`), not on this branch — check out `rl_yifei_clean` to
run them. They submit through the same aifsdk `submit_job.sh` path and the
same `bonete61` conventions as the launchers above. Every `submit_*.sh` has a
matching `run_*.sh` in-pod driver that is exec'd as the pod `--cmd` and never
invoked by hand; all knobs are env vars documented in each script's header.
Defaults assume the author's dev box (`/data/t-yifeili/...`); override
`SUBMIT`/`AIFSDK_ROOT`, `CREDENTIALS_FILE`, and PVC alias paths for your own
layout.

### SFT training (LlamaFactory full-SFT, 8xB200)

| Command | Purpose |
| --- | --- |
| `bash docker/upload_data_to_pvc.sh <local-dir> [dest-name]` | Upload a data bundle to the fixed PVC path `/mnt/pvc/experiments/<alias>/data/<name>` (chunked, verified, cross-user readable). |
| `CONFIG=examples/train_full/<cfg>.yaml bash docker/submit_sft_q35_image.sh` | Submit a train-to-completion SFT job. `NODES`/`GPUS` for multi-node DP; `RESUME_FROM_CKPT` + `TARGET_TOTAL_EPOCHS` for warm restart (same `NODES` as the original run). |
| `SFT_CONFIG=examples/train_full/<cfg>.yaml bash docker/submit_sft_eval_q35_image.sh` | Combined train-then-eval job: SFT, then on success a multi-node harness eval of the trained checkpoint on the same nodes (`EVAL_BACKEND=harness` default). |

Canonical chain: `upload_data_to_pvc.sh` → point the yaml's
`dataset_dir`/`media_dir` at the PVC path → submit. With `LIGHT_UPLOAD=1`
(default) the submit auto-runs `make_light_code_tree.sh` (slim code upload)
and sources `wandb_key_preflight.sh` (refuses to submit without a valid
`WANDB_API_KEY`). Warm restarts run `prepare_warm_restart.py` in-pod, which
calls `scripts/merge_vision_from_base.py` to restore the vision tower a
language-only Qwen3.5 checkpoint is missing.

### Distributed OM2W eval (multi-node data-parallel, resumable)

| Command | Purpose |
| --- | --- |
| `EVAL_CKPT=/mnt/pvc/<alias>/models/... NODES=4 bash docker/submit_dist_eval_q35_image.sh` | Shard the 300 OM2W tasks across nodes; each node serves the checkpoint with a tp=8 vLLM, the master judges and aggregates. Resubmitting the same `EVAL_RUN_ID` resumes; `RETRY_FAILED=1` re-runs failed tasks. |
| `bash docker/fetch_eval_rollouts.sh <JOB\|short-id> [dest]` | Stream a job's rollout artifacts from the (pod-only) PVC down to the dev box via a throwaway CPU pod. |

Key knobs: `TASK_LEVEL` (`all`), `TOTAL_WORKERS` (80 — the verified
Browserbase concurrency ceiling across `NODES × WORKERS`; do not raise
without checking quota), `JUDGE_RUNS`/`JUDGE_MODEL`/`JUDGE_ENDPOINT`,
`REQUIRE_RUNTIME_MANIFEST=1` for PhiTrain WebWright checkpoints. Outputs land
at `$DATA_ROOT/evals/$EVAL_RUN_ID/outputs` — the input to every judge path
below.

### Judge-only passes (CPU, 0 GPU, over existing trajectories)

| Command | Purpose |
| --- | --- |
| `EVAL_RUN_ID=<id> bash docker/submit_judge_only_q35_image.sh` | Mode C: original OM2W WebJudge over an existing trajectory dir; resumable by `task_id`; usable while the generation job is still running. |
| `docker/run_pcli_judge.sh` | In-pod runner for the persistent-CLI **steps** judge (`TRAJ_DIR`/`OUT_DIR` env); submitted ad hoc as a pod `--cmd`, see `docs/PIPELINE_SPB.md`. |
| `docker/run_step_script_judge.sh` | In-pod mode-D driver: layout preflight, then the step-script judge at 150 workers; gateway-only key wiring. Submitted ad hoc. |
| `python scripts/eval_with_original_om2w.py --trajectories_dir <dir> --output_path <dir>` | Base vendored WebJudge over a mini-web-agent output dir (`final_runs`/`steps` layouts). |
| `python scripts/eval_persistent_cli_with_original_om2w.py ...` | Judge adapter for persistent-browser runs (`browser-steps.jsonl` + `screenshots/`). |
| `python scripts/eval_persistent_cli_steps_with_original_om2w.py ...` | Same, judging from stored low-level actions (`steps/*.sh` or `result.json` histories). |
| `python scripts/eval_om2w_full_context.py --trajectories_dir <dir> --output_path <dir>` | Full-context judge (screenshots + actions + thoughts) so base-model trajectories without SFT-style action lines still get judged. |
| `python scripts/spb_level_breakdown.py --results-dir <judge-out> --tasks-file src/miniswewebagent/run/benchmarks/om2w_260220.json` | Total + per-level success from a judge results dir. |

All judge scripts are resumable by re-pointing at the same output path, and
all load the vendored judge through a `/home/luyadong/sandbox/mini-web-agent`
symlink that each cluster runner creates first.

### Mini-harness eval (single node, historical harness)

| Command | Purpose |
| --- | --- |
| `bash docker/submit_mini_harness_eval_q35_image.sh` | Single-node eval pod with the historical `miniswewebagent` harness (no sharding/resume; superseded by the dist eval for full runs). |
| `CKPT=<hf-ckpt> bash scripts/mini_harness_eval_sft_vllm.sh` | Local counterpart: serve the checkpoint with vLLM and run the harness; `START_VLLM=0 ENDPOINT=...` reuses a running server; `SMOKE=1` for a 1-task sanity pass. |

### ScienceAgentBench (SAB)

| Command | Purpose |
| --- | --- |
| `python scripts/make_sab_tasks_json.py --out src/miniswewebagent/run/benchmarks/sab_verified.json` | One-time: build the SAB (verified) tasks file with a training-aligned prompt. |
| `CKPT=<ckpt> bash scripts/sab_eval_sft_vllm.sh` | Stage A: serve + generate one program per task into `pred_programs/`; `GEN_MODE=singleturn` (default, via `sab_infer_singleturn.py`) or `agent` (om2w harness + `collect_sab_preds.py`). Prints the Stage B scoring command to run inside the ScienceAgentBench repo. |

### Checkpoint / storage utilities

| Command | Purpose |
| --- | --- |
| `python docker/merge_skyrl_ckpt_to_hf.py --policy-dir <ckpt>/policy --base <hf-dir> --out <dir>` | CPU-only merge of a SkyRL FSDP2 sharded checkpoint into HF safetensors (needs a ≥128 GB-RAM node for 9B). |
| `OUT_BASE=... BASE_HF=... MERGER=... bash docker/watch_ckpt_export_hf.sh` | Poll loop that HF-exports each new SkyRL `global_step_N` before the trainer prunes it. |
| `bash scripts/az_ckpt.sh sas\|push\|pull` | Move multi-GB checkpoints between pods and dev boxes via Azure Blob (azcopy engine; workload-identity or SAS auth). |
| `SET=easy\|medium\|hard\|full bash scripts/eval_online_m2w_local_4gpu.sh` | Local 4-GPU SkyRL `eval_entrypoint` OM2W eval of an SFT checkpoint (SFT-aligned prompt/parser). |

## Environment conventions

- `REPO` overrides automatic repository-root discovery for local scripts.
- `CREDENTIALS_FILE` selects the credential file used by the local runner;
  `CRED_FILE` remains a legacy alias.
- `VENV_BIN` selects the runner virtual environment's `bin` directory.
- Cluster launchers list their supported environment overrides in `--help`.
- Script-specific model, endpoint, output, and worker defaults are documented
  in each entry point's header or help output.
- The tunnel helper prefers `cloudflared` on `PATH`; otherwise it downloads the
  pinned 2026.3.0 binary into the user cache and verifies the GitHub release
  checksum. `CLOUDFLARED_BIN` selects an existing custom binary.
