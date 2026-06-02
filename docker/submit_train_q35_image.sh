#!/usr/bin/env bash
# Launch a NON-INTERACTIVE TRAINING job on the *generic* qwen3.5 image
#   aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415
# on a single 8xB200 node; runs to completion (no sleep-infinity debug pod).
#
# Default config = Qwen3.5-9B on 8 GPUs (echo_configs/qwen35_9b_web_agent_easy_8gpu.yaml,
# which sets distributed_executor_backend=mp so the 8 vLLM engines spread across
# GPU 0-7 instead of piling on GPU 0). Override with CONFIG=.
#
# All setup + the training launch live in the uploaded driver
# (docker/run_train_q35_image.sh); `--cmd` is just a tiny exec of it, keeping the
# `kubectl create` POST body clear of the bonete61 Cloudflare WAF that blocks big
# inline shell preambles. Submitted with --follow-logs so it tails the run.
#
#   bash docker/submit_train_q35_image.sh
#   CONFIG=echo_configs/qwen35_4b_web_agent_easy_4gpu.yaml GPUS=4 bash docker/submit_train_q35_image.sh
#
# Kill manually with:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
GPUS="${GPUS:-8}"
# Config path is relative to SkyRL/ (echo_configs lives inside the uploaded SkyRL).
CONFIG="${CONFIG:-echo_configs/qwen35_9b_web_agent_easy_8gpu.yaml}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done
[[ -f "$SKYRL_DIR/$CONFIG" ]] || { echo "[error] config not found: $SKYRL_DIR/$CONFIG"; exit 1; }

export PATH="$HOME/.krew/bin:$PATH"

# WandB host. submit_job.sh defaults WANDB_HOST to microsoft-research.wandb.io,
# which 401s the personal public-wandb WANDB_API_KEY. Force the public host so the
# host-shell WANDB_API_KEY routes to https://api.wandb.ai. Override at call time
# for MS-internal: WANDB_HOST=https://microsoft-research.wandb.io ...
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"

# Job priority. PRIORITY is a naming label in JOB_NAME (the GPU monitor dashboard
# buckets on it). PRIORITY_CLASS_NAME is the REAL k8s PriorityClass and MUST be
# one of high/medium/low -- 'p0' is NOT a valid class, so p0 maps to 'high'.
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_train_q35_image] GPUS=$GPUS IMAGE=$IMAGE CONFIG=$CONFIG"
echo "[submit_train_q35_image] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME WANDB_HOST=$WANDB_HOST"

# Tiny --cmd: just exec the uploaded driver. Creds come from the two secret
# volumes; TRAIN_CONFIG is forwarded via --extra-env-vars. No raw secret and no
# big preamble in the request body, so WAF-safe.
bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --secret-volume echo-rl-openai:/run/secrets/echo-rl-openai \
    --extra-env-vars "TRAIN_CONFIG=${CONFIG}" \
    --follow-logs \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_train_q35_image.sh'
