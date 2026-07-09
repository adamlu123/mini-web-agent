---
name: web-agent-sft-cluster
description: >-
  Train a Qwen3.5 web-agent SFT model from rollout trajectories and evaluate it on the
  bonete (lambda / bonete61, NVIDIA B200) cluster. Use when the user wants to: convert
  mini-web-agent online rollout trajectories into LlamaFactory ShareGPT SFT data, merge
  datasets, submit a combined train→eval job, or run cluster eval of an SFT checkpoint.
  Covers data prep, dataset registration, the ZeRO-2 9B SFT config, vision-tower merge,
  the SINGLE-ENGINE + SFT-ALIGNED eval setup (and why), and job monitoring.
---

# Web-agent SFT + eval on the bonete cluster

End-to-end pipeline: turn web-agent rollout trajectories into a LlamaFactory SFT
dataset, full-finetune Qwen3.5 (4B or 9B) on it, and evaluate the trained
checkpoint on the om2w-easy web tasks — all on the bonete61 / lambda B200 cluster.

Everything lives in `/data/t-yifeili/mini-web-agent`. Run all commands from there
unless noted. The driver/submit scripts are under `docker/`.

## TL;DR — the happy path

```bash
cd /data/t-yifeili/mini-web-agent

# 1. trajectories -> sharegpt SFT json (one folder of <task>/trajectory.json per src)
python LlamaFactory/scripts/make_web_agent_sft.py \
  --src /path/to/<run>/<...>_success \
  --out LlamaFactory/data/web_agent_<tag>.json \
  --max-obs-chars 0                      # 0 = untruncated obs (matches *_full datasets)

# 2. (optional) merge with an existing dataset, then register + make a train config
#    (see "Step 2/3" below). Register the merged json in LlamaFactory/data/dataset_info.json.

# 3. submit ONE job that trains then evals (9B defaults, rlscaling workstream):
PROJECT_NAME=rlscaling bash docker/submit_sft_eval_q35_image.sh

# ...or eval an already-trained ckpt on its own (clean pod, reads the PVC ckpt):
PROJECT_NAME=rlscaling \
EVAL_CKPT=/mnt/pvc/t-yifeili/models/qwen35_9b/full/websft_merged \
  bash docker/submit_eval_q35_image.sh
```

The submit's `--follow-logs` **detaches while the pod keeps running** (it prints
`Pod final phase: Running` and exits 0). That is NOT a failure — follow the live
job with `kubectl logs -f` instead (see "Monitoring").

## Prerequisites

- Repos uploaded by the submit scripts: `mini-web-agent` (LlamaFactory lives
  inside it) and, for eval, `SkyRL` (`/data/t-yifeili/SkyRL`).
- Cluster submit driver: `/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh`.
- `kubectl` + krew on PATH: `export PATH="$HOME/.krew/bin:$PATH"`; namespace `bonete61`.
- Secrets mounted into the pod as volumes (already provisioned):
  - `echo-rl-creds` → `/run/secrets/echo-rl-creds/cred.sh` (HF token, Browserbase).
  - `echo-rl-openai` → `/run/secrets/echo-rl-openai/OPENAI_API_KEY` (working OSW-judge key).
- Image: `aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415`
  (the generic qwen3.5 image; bakes torch 2.10 / vllm 0.18 that actually run qwen3_5).
- **Workstream**: pass `PROJECT_NAME=rlscaling` to avoid queueing in `cua`.

## Step 1 — trajectories → ShareGPT SFT data

`LlamaFactory/scripts/make_web_agent_sft.py` converts a directory of
`<task>/trajectory.json` (the online gpt-5.x rollout format: `messages` with
role/content/extra, `environment.config.output_dir`) into a multi-turn ShareGPT
dataset. Per assistant turn the target is
`<think>…</think><bash>…</bash>` (or `…</think><answer>…</answer>` to finish).
It scrubs operator paths → `/workspace`, redacts secrets, and (with
`--max-obs-chars 0`) keeps observations untruncated.

```bash
python LlamaFactory/scripts/make_web_agent_sft.py \
  --src /data/t-yifeili/0601/0601/N500_s100_agnostic_r2_success \
  --out LlamaFactory/data/web_agent_n500_r2_success.json \
  --max-obs-chars 0
```

Trajectories with only `system`+`user` (no assistant turn) are dropped — that's
expected, not an error.

## Step 2 — merge datasets + register

Merging is plain JSON concatenation of the sharegpt arrays:

```bash
python - <<'PY'
import json
old = json.load(open('LlamaFactory/data/web_agent_pae100_full.json'))   # 86 convos
new = json.load(open('LlamaFactory/data/web_agent_n500_r2_success.json'))# 142 convos
merged = old + new
json.dump(merged, open('LlamaFactory/data/web_agent_pae_merged.json','w'),
          ensure_ascii=False, indent=2)
g = sum(1 for ex in merged for t in ex['conversations'] if t['from']=='gpt')
print(len(merged), 'convos,', g, 'assistant turns')
PY
```

Register the file in `LlamaFactory/data/dataset_info.json` (copy an existing
`web_agent_*` entry — sharegpt formatting, `conversations`/`system` columns,
`from`/`value`/`human`/`gpt` tags):

```json
"web_agent_pae_merged": {
  "file_name": "web_agent_pae_merged.json",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations", "system": "system" },
  "tags": { "role_tag": "from", "content_tag": "value",
            "user_tag": "human", "assistant_tag": "gpt", "observation_tag": "observation" }
}
```

## Step 3 — training config

Copy an existing `LlamaFactory/examples/train_full/qwen35_{4b,9b}_websft.yaml`,
point `dataset:` at the new name, and set a fresh `output_dir:`.

- **9B uses ZeRO-2, NOT ZeRO-3** (`deepspeed: examples/deepspeed/ds_z2_config.json`).
  z3's `zero.init()` fails to load pretrained weights into this Qwen3.5
  VL/ImageTextToText model → trains from RANDOM init (step-1 loss ≈ ln(248k) ≈
  12.4). z2 loads weights normally; 9B fits on a 180 GB B200. (4B uses z3+offload.)
- `template: qwen3_5`, `mask_history: false`, `enable_thinking: true` (loss on
  every assistant turn including its `<think>`).
- `cutoff_len: 32768` — untruncated obs make some convos exceed this and get
  tail-truncated (a few `<answer>` may be lost). Raise it, or regenerate the data
  with a smaller `--max-obs-chars`, if that matters.

Healthy 9B run: 228 convos → 29 steps, ~22 min on 8 GPUs, step-1 loss ≈ 0.77
(NOT ~12.4), final train_loss ≈ 0.46.

## Step 4 — submit (train→eval in one job)

```bash
PROJECT_NAME=rlscaling bash docker/submit_sft_eval_q35_image.sh
# overridable:
PROJECT_NAME=rlscaling \
SFT_CONFIG=examples/train_full/qwen35_9b_websft_merged.yaml \
EVAL_CONFIG=configs/qwen35_9b_web_agent_easy_eval_sft.yaml \
EVAL_RUN_TAG=merged_9b \
  bash docker/submit_sft_eval_q35_image.sh
```

What it does (`docker/run_sft_q35_image.sh` in-pod):
1. Installs LlamaFactory + the few deps the image lacks, runs `llamafactory-cli
   train $SFT_CONFIG` under torchrun on all GPUs.
2. On success, **syncs** the HF ckpt to a stable PVC path that survives future
   jobs: `/mnt/pvc/$USER/models/<output_dir without saves/>`
   (e.g. `…/models/qwen35_9b/full/websft_merged`).
3. **Merges the vision tower** back from the base (LlamaFactory text-SFT on
   qwen3_5 drops/mis-prefixes the vision keys, so the raw ckpt can't reload as
   `Qwen3_5ForConditionalGeneration`). `MERGE_VISION=0` disables.
4. If `EVAL_AFTER=1` (set by this submit), runs the cluster eval driver on that
   ckpt in the SAME pod.

Uploads BOTH `mini-web-agent` and `SkyRL` and mounts BOTH secret volumes (the
eval phase needs the full SkyRL + RL/eval stack).

## Step 5 — eval an existing checkpoint (standalone, clean pod)

```bash
PROJECT_NAME=rlscaling \
EVAL_CKPT=/mnt/pvc/t-yifeili/models/qwen35_9b/full/websft_merged \
EVAL_CONFIG=configs/qwen35_9b_web_agent_easy_eval_sft.yaml \
EVAL_RUN_TAG=merged_9b_sft \
  bash docker/submit_eval_q35_image.sh
```

`docker/run_eval_q35_image.sh` is the cluster equivalent of the local
`run_local_eval.sh`: bootstraps the RL/eval stack (same as the RL train driver),
loads the HF ckpt into vLLM, and runs a single `evaluate()` pass over the om2w
easy train+val parquets (each scored separately), logging
`eval/web_agent_om2w_easy_{train,val}/avg_score|pass_at_1` to the console. On
failure it dumps the SkyRL infra log + ray worker tracebacks (the real
EngineCore error does NOT reach pod stdout otherwise).

`EVAL_CKPT` must be a path visible INSIDE the pod (the PVC). The driver also
exports `WEB_SFT_CKPT=$EVAL_CKPT` so the SFT-aligned config resolves the ckpt's
tokenizer / chat_template / weights.

## Monitoring

`--follow-logs` detaches while the pod runs. Follow the live job directly, with
auto-reconnect and a clean stop on the terminal phase:

```bash
export PATH="$HOME/.krew/bin:$PATH"
POD=t-yifeili-p0-rlscaling-job-XXXXX-master-0   # from "Created job:"; append -master-0
NS=bonete61
while true; do
  kubectl -n $NS logs -f --tail=20 $POD 2>&1 | grep -E --line-buffered \
    "train_runtime|train_loss|\[sync\] OK|\[merge\] wrote|EVAL_AFTER|Evaluation Progress|Metrics from web-agent|eval/web_agent_om2w|exited rc=|No available memory|out of memory|ChildFailedError"
  p=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null)
  case "$p" in Succeeded|Failed|"") echo "phase=$p"; break;; esac; sleep 8
done
```

- Pull the full log: `kubectl -n bonete61 logs <pod> > out.log`.
- Kill a job: `kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false`.
- WARNING when grepping logs for failures: the web-agent training DATA and eval
  rollouts contain `Traceback` / `Error` / `Timeout` / `playwright` strings.
  Grep for STRUCTURED job signals (`'loss':`, `[sync]`, `[eval]`, `exited rc=`,
  `No available memory`, `ChildFailedError`), not bare `error`/`Traceback`.

## GOTCHAS — read before debugging eval

These are the traps that cost real jobs; the scripts now bake in the fixes.

1. **Eval must use the SFT-ALIGNED config for an SFT-trained model.** The base
   eval config (`configs/qwen35_9b_web_agent_easy_eval.yaml`) uses the qwen35
   **tool-call** prompt/parser (`<tool_call><function=bash>…`). SFT data trains
   the `<think>/<bash>/<answer>` format. Mismatch ⇒ thousands of parse/format
   violations and a near-zero score that LOOKS like a bad model. Use
   `configs/qwen35_9b_web_agent_easy_eval_sft.yaml` (`parser_name: bash`,
   `prompt_mode: sft`, `env_overrides.sft_mode: true`; tokenizer/chat_template/
   model from `${WEB_SFT_CKPT}`). This is the default in the submit scripts now.
2. **Eval needs `colocate_all=false`** (the driver forces it). With
   colocate_all=true SkyRL sleeps the vLLM engine at level=2 expecting an FSDP
   weight-sync that never comes in eval-only → corrupted weights → all scores 0.
3. **Single vLLM engine for eval.** With `colocate_all=false` + `num_engines=4`,
   Ray PILES all 4 engines onto GPU 0 on the cluster (locally on 4 GPUs they
   spread 1/GPU and it "just works") → CUDA OOM on GPU 0 while GPUs 1-3 idle. The
   driver overrides to `num_engines=1, tensor_parallel_size=1, enforce_eager=true,
   gpu_memory_utilization=0.85, max_num_batched_tokens=32768` (one 9B engine fits
   with room to spare on one 180 GB B200; eval only scores ~80 tasks). Override
   via `EVAL_ENGINE_OVERRIDES` (space-separated dotlist; empty string = none).
4. **Don't trust "Engine core initialization failed. Failed core proc(s): {}".**
   That's the downstream message; the real cause (e.g.
   `ValueError: No available memory for the cache blocks`, `torch.OutOfMemoryError`)
   is in the EngineCore subprocess / SkyRL infra log. The eval driver dumps those
   on failure.
5. **9B → ZeRO-2** (see Step 3). Wrong DeepSpeed stage silently trains garbage.
6. **`--follow-logs` exiting ≠ job done** (see Monitoring).

## Key files

| Path | Role |
|------|------|
| `LlamaFactory/scripts/make_web_agent_sft.py` | trajectory.json → sharegpt SFT json |
| `LlamaFactory/data/dataset_info.json` | register datasets here |
| `LlamaFactory/examples/train_full/qwen35_9b_websft_merged.yaml` | 9B merged-data train config (ZeRO-2) |
| `docker/submit_sft_eval_q35_image.sh` | submit combined train→eval (uploads both repos) |
| `docker/run_sft_q35_image.sh` | in-pod: train + sync + vision-merge + (EVAL_AFTER) eval |
| `docker/submit_eval_q35_image.sh` | submit standalone eval of a ckpt |
| `docker/run_eval_q35_image.sh` | in-pod cluster eval (single-engine, failure diag) |
| `configs/qwen35_9b_web_agent_easy_eval_sft.yaml` | SFT-aligned eval config (use this!) |
| `run_local_eval.sh` | the LOCAL (4×H100) eval the cluster driver mirrors |
| `/mnt/pvc/$USER/models/<...>` | stable PVC ckpt path (survives future jobs) |
