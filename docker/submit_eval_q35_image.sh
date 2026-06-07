#!/usr/bin/env bash
# Submit an EVAL-ONLY cluster job on the generic qwen3.5 image, single node.
# Cluster-side equivalent of run_local_eval.sh: runs docker/run_eval_q35_image.sh
# over the om2w easy train+val parquets (OSW judge), either on the base weights
# from the eval config or on a trained HF ckpt you pass via EVAL_CKPT.
#
# Uploads BOTH mini-web-agent AND SkyRL (the eval needs the full RL/eval stack)
# and mounts both secret volumes, exactly like the training submit.
#
# Usage:
#   # base weights from the config
#   bash docker/submit_eval_q35_image.sh
#   # a trained HF ckpt that already lives on the cluster PVC
#   EVAL_CKPT=/mnt/.../models/qwen35_4b/full/websft_merged \
#     bash docker/submit_eval_q35_image.sh
#
# NOTE: EVAL_CKPT must be a path visible INSIDE the pod (e.g. on the PVC). To
# eval a ckpt produced by a prior training job, point it at that job's synced
# model dir ($PVC_MOUNT/$USER_ALIAS/models/...). For a fresh train->eval in one
# shot use docker/submit_sft_eval_q35_image.sh instead.
#
# Kill manually with:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
GPUS="${GPUS:-4}"   # eval config uses 4 vllm engines / 4 GPUs
EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen35_9b_web_agent_easy_eval_sft.yaml}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-base_9b}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done
[[ -f "$MINI_WEB_AGENT_DIR/$EVAL_CONFIG" ]] || { echo "[error] eval config not found: $EVAL_CONFIG"; exit 1; }

export PATH="$HOME/.krew/bin:$PATH"
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_eval_q35_image] GPUS=$GPUS IMAGE=$IMAGE EVAL_CONFIG=$EVAL_CONFIG"
echo "[submit_eval_q35_image] EVAL_CKPT=${EVAL_CKPT:-<base weights from config>} TAG=$EVAL_RUN_TAG"

EXTRA_ENV="EVAL_CONFIG=${EVAL_CONFIG},EVAL_RUN_TAG=${EVAL_RUN_TAG}"
[[ -n "${EVAL_CKPT:-}" ]] && EXTRA_ENV="${EXTRA_ENV},EVAL_CKPT=${EVAL_CKPT}"

bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --secret-volume echo-rl-openai:/run/secrets/echo-rl-openai \
    --extra-env-vars "$EXTRA_ENV" \
    --follow-logs \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_eval_q35_image.sh'
