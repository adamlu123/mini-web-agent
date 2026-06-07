#!/usr/bin/env bash
# Submit a COMBINED "train then eval" job on the generic qwen3.5 image, single
# 8xB200 node, runs to completion:
#   1. LlamaFactory full-SFT (docker/run_sft_q35_image.sh) on $SFT_CONFIG, then
#   2. on success, a cluster eval (docker/run_eval_q35_image.sh) of the trained
#      HF ckpt over the om2w easy train+val parquets (OSW judge).
#
# Unlike submit_sft_q35_image.sh (SFT-only, uploads just mini-web-agent), this
# uploads BOTH mini-web-agent AND SkyRL and mounts BOTH secret volumes, because
# the eval phase needs the full SkyRL + RL/eval stack (vllm rollout, browserbase,
# OSW judge). The SFT phase ignores them; the eval phase (EVAL_AFTER=1) bootstraps
# the RL stack on top of the LlamaFactory env inside the same pod.
#
# Usage:
#   bash docker/submit_sft_eval_q35_image.sh
#   SFT_CONFIG=examples/train_full/qwen35_4b_websft_merged.yaml \
#   EVAL_CONFIG=configs/qwen35_4b_web_agent_easy_eval.yaml \
#     bash docker/submit_sft_eval_q35_image.sh
#
# Kill manually with:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
GPUS="${GPUS:-8}"
# Train config: path relative to LlamaFactory/ (lives inside the uploaded repo).
SFT_CONFIG="${SFT_CONFIG:-examples/train_full/qwen35_9b_websft_merged.yaml}"
# Eval config: path relative to mini-web-agent/ root.
EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen35_9b_web_agent_easy_eval_sft.yaml}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-merged_9b}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done
[[ -f "$MINI_WEB_AGENT_DIR/LlamaFactory/$SFT_CONFIG" ]] || { echo "[error] SFT config not found: LlamaFactory/$SFT_CONFIG"; exit 1; }
[[ -f "$MINI_WEB_AGENT_DIR/$EVAL_CONFIG" ]] || { echo "[error] eval config not found: $EVAL_CONFIG"; exit 1; }

export PATH="$HOME/.krew/bin:$PATH"
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-rlscaling}"

echo "[submit_sft_eval_q35_image] GPUS=$GPUS IMAGE=$IMAGE"
echo "[submit_sft_eval_q35_image] SFT_CONFIG=$SFT_CONFIG"
echo "[submit_sft_eval_q35_image] EVAL_CONFIG=$EVAL_CONFIG (EVAL_AFTER=1, tag=$EVAL_RUN_TAG)"
echo "[submit_sft_eval_q35_image] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME"

# Forward both phases' knobs. EVAL_AFTER=1 flips on the eval chaining inside
# run_sft_q35_image.sh after a successful train + ckpt sync.
EXTRA_ENV="SFT_CONFIG=${SFT_CONFIG},NPROC=${GPUS},EVAL_AFTER=1,EVAL_CONFIG=${EVAL_CONFIG},EVAL_RUN_TAG=${EVAL_RUN_TAG}"
[[ -n "${AZBLOB_AUTO_PUSH:-}" ]] && EXTRA_ENV="${EXTRA_ENV},AZBLOB_AUTO_PUSH=${AZBLOB_AUTO_PUSH}"
[[ -n "${AZBLOB_SAS_TOKEN:-}" ]] && EXTRA_ENV="${EXTRA_ENV},AZBLOB_SAS_TOKEN=${AZBLOB_SAS_TOKEN}"
[[ -n "${AZBLOB_PREFIX:-}" ]]    && EXTRA_ENV="${EXTRA_ENV},AZBLOB_PREFIX=${AZBLOB_PREFIX}"

# Tiny --cmd execs the uploaded SFT driver (which chains the eval driver when
# EVAL_AFTER=1). Both secret volumes mounted: echo-rl-creds (HF + browserbase)
# and echo-rl-openai (working judge key). WAF-safe.
bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --secret-volume echo-rl-openai:/run/secrets/echo-rl-openai \
    --extra-env-vars "$EXTRA_ENV" \
    --follow-logs \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_sft_q35_image.sh'
