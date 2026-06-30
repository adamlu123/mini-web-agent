# Web-Agent Label1 4-Node SFT Run 2026-06-30

## Purpose

Run a full Qwen3.5-9B SFT job on the new self-judge-label1 web-agent auxiliary dataset with 40k context, using 4 bonete nodes to increase the effective global batch size without increasing per-GPU batch size.

## Active Run

- Volcano job: `t-yifeili-p0-cua-job-7fc3a`
- Namespace: `bonete61`
- Workstream / project bucket: `cua`
- Priority bucket: `p0`
- Kubernetes priority class: `high`
- Queue: `bonete61`
- Status at last check: `Running`
- W&B project: `https://wandb.ai/flyhero99/web-agent-sft`
- W&B run: `https://wandb.ai/flyhero99/web-agent-sft/runs/skgcb842`

## Pod Layout

Volcano task layout:

```text
master replicas=1
worker replicas=3
```

Current pod placement:

```text
t-yifeili-p0-cua-job-7fc3a-master-0   Running   slc01-cl02-hgx-0322   10.45.219.152
t-yifeili-p0-cua-job-7fc3a-worker-0   Running   slc01-cl02-hgx-0176   10.45.83.208
t-yifeili-p0-cua-job-7fc3a-worker-1   Running   slc01-cl02-hgx-0122   10.44.90.172
t-yifeili-p0-cua-job-7fc3a-worker-2   Running   slc01-cl02-hgx-0455   10.45.38.154
```

The pod driver reports:

```text
NNODES=4
NODE_RANK=0 on master
NPROC=8 per node
```

## Training Config

Primary config:

```text
LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_label1_allwebchain_sr1k_40k_4node_pvcdata.yaml
```

The config reads data from an existing PVC upload to avoid re-uploading the 5 GB portable dataset bundle:

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
finetuning_type: full
deepspeed: examples/deepspeed/ds_z2_config.json
use_reentrant_gc: false
enable_liger_kernel: false

dataset_dir: /mnt/pvc/t-yifeili/runs/t-yifeili-p0-cua-job-af8b0/mini-web-agent/LlamaFactory/data/web_agent_seq_sessions_tools_label1_allwebchain_sr1k_0629_portable_bundle
media_dir: /mnt/pvc/t-yifeili/runs/t-yifeili-p0-cua-job-af8b0/mini-web-agent/LlamaFactory/data/web_agent_seq_sessions_tools_label1_allwebchain_sr1k_0629_portable_bundle
dataset: web_agent_seq_sessions_tools_label1_allwebchain_sr1k_0629_portable

cutoff_len: 40000
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
num_train_epochs: 2.0
learning_rate: 1.0e-5
warmup_ratio: 0.05
bf16: true
report_to: wandb
```

Effective global batch:

```text
1 per-device batch * 1 grad accumulation * 4 nodes * 8 GPUs = 32
```

Trainer initialization confirmed:

```text
Num examples = 9,360
Num Epochs = 2
Num update steps per epoch = 293
Total train batch size = 32
Total optimization steps = 586
```

## Dataset

Source generation script:

```text
LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py
```

Source dataset committed via Git LFS:

```text
LlamaFactory/data/web_agent_seq_sessions_tools_label1_allwebchain_sr1k_0629.json
```

Portable image bundle used by this run:

```text
LlamaFactory/data/web_agent_seq_sessions_tools_label1_allwebchain_sr1k_0629_portable_bundle
```

Portable bundle manifest:

```text
examples: 9367
image_references: 7379
unique_images_copied: 6233
missing_images: 0
local bundle size: about 5.0G
```

The source dataset contains only self-judge-label1 trajectories from:

```text
/data/t-yifeili/sft_data/pae_100
/data/t-yifeili/0601/0601/N500_s100_agnostic_r2_success
/data/t-yifeili/webchain_sampling/outputs
```

Sample counts in the capped dataset:

```text
trajectory_session: 6379
image_qa: 988
self_reflection_image: 1000
self_reflection_final: 1000
total examples: 9367
```

## Submit Command

This run avoids another 5 GB upload by submitting a lightweight code tree and pointing the config at the PVC dataset path already uploaded by the earlier `af8b0` attempt.

```bash
cd /data/t-yifeili/mini-web-agent

WANDB_HOST=https://api.wandb.ai \
WANDB_PROJECT=web-agent-sft \
PRIORITY=p0 \
PROJECT_NAME=cua \
PRIORITY_CLASS_NAME=high \
bash /data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh \
  --upload /tmp/bonete-sft-smoke/mini-web-agent \
  --image aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415 \
  --node 4 \
  --gpu-per-node 8 \
  --cpu 64 \
  --memory 512Gi \
  --shm 64Gi \
  --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
  --extra-env-vars 'SFT_CONFIG=examples/train_full/qwen35_9b_web_agent_seq_label1_allwebchain_sr1k_40k_4node_pvcdata.yaml,NPROC=8,AZBLOB_AUTO_PUSH=0' \
  --follow-logs \
  --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_sft_q35_image.sh'
```

## Prior Failed Attempts

- `t-yifeili-p0-cua-job-af8b0`: reached trainer initialization but failed before first loss because W&B was pointed at the Microsoft endpoint and returned 401.
- `t-yifeili-p0-cua-job-b47dc`: retry with personal W&B endpoint failed during the 4.3 GB tar upload with connection reset.
- CPU data-copy attempt `t-yifeili-p3-datacopy-013704` was stopped because `cp -a` on Vast PVC emitted many permission-preservation errors. This was no longer needed because the `af8b0` upload had already produced a usable portable bundle path.

## Monitoring

```bash
export PATH="$HOME/.krew/bin:$PATH"
NS=bonete61
JOB=t-yifeili-p0-cua-job-7fc3a
POD=${JOB}-master-0

kubectl -n $NS logs -f --tail=50 $POD | grep -E --line-buffered \
  "Num examples|Num Epochs|Total train batch size|Total optimization steps|loss|train_loss|wandb|SFT exited rc=|OutOfMemory|CUDA out of memory|ChildFailedError|Saving model|\\[sync\\]"
```

## Current Notes

- W&B login now uses the personal endpoint successfully: `https://api.wandb.ai`.
- W&B user/project resolved to `flyhero99/web-agent-sft`.
- As of the recorded check, no OOM or `ChildFailedError` had appeared.
- At the recorded check, the job had entered trainer initialization and W&B sync, but no first loss line had been observed yet.