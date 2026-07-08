#!/usr/bin/env bash
# Submit a LlamaFactory full-SFT job on the generic qwen3.5 image, single 8xB200
# node, runs to completion. Much lighter than the RL submits: uploads only
# mini-web-agent (LlamaFactory lives inside it), no SkyRL, no browserbase/openai.
#
# Default CONFIG = the SMOKE yaml (cheap 4-GPU-style validation of the cluster
# path on mock data). Switch to the full yaml + your real dataset once the
# trajectory->sharegpt conversion is done:
#   CONFIG=examples/train_full/qwen35_4b_websft.yaml bash docker/submit_sft_q35_image.sh
#
# NOTE: no `--azblob` -> submit_job.sh tars with `--exclude-vcs-ignores`, so the
# gitignored `outputs/` (large local run artifacts) is NOT uploaded.
#
# Kill manually with:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
GPUS="${GPUS:-8}"
# NODES>1 -> multi-node data-parallel. submit_job.sh wires the volcano `pytorch`
# plugin (MASTER_ADDR/PORT, WORLD_SIZE, RANK); run_sft_q35_image.sh maps those to
# NNODES/NODE_RANK for the LlamaFactory torchrun launcher and master-guards the
# code rsync + ckpt sync. Global batch scales with total GPUs (NODES*GPUS).
NODES="${NODES:-1}"
# Config path is relative to LlamaFactory/ (which lives inside the uploaded repo).
CONFIG="${CONFIG:-examples/train_full/qwen35_4b_websft_smoke.yaml}"
# LIGHT_UPLOAD=1 (default): rsync a code tree WITHOUT LlamaFactory/data/web_agent_*
# to /tmp and upload that. Training data must already live at the fixed PVC path
# /mnt/pvc/experiments/t-yifeili/data/<bundle> (docker/upload_data_to_pvc.sh) and
# the yaml's dataset_dir/media_dir must point there. Set LIGHT_UPLOAD=0 to upload
# the full working tree (old behavior, multi-GB, flaky).
LIGHT_UPLOAD="${LIGHT_UPLOAD:-1}"

[[ -d "$MINI_WEB_AGENT_DIR" ]] || { echo "[error] missing $MINI_WEB_AGENT_DIR"; exit 1; }
[[ -f "$MINI_WEB_AGENT_DIR/LlamaFactory/$CONFIG" ]] || { echo "[error] config not found: LlamaFactory/$CONFIG"; exit 1; }

if [[ "$LIGHT_UPLOAD" == "1" ]]; then
    MINI_WEB_AGENT_DIR="$(SRC="$MINI_WEB_AGENT_DIR" bash "$(dirname "$0")/make_light_code_tree.sh" | tail -1)"
    echo "[submit_sft_q35_image] LIGHT_UPLOAD=1 -> uploading $MINI_WEB_AGENT_DIR"
fi

export PATH="$HOME/.krew/bin:$PATH"
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_sft_q35_image] NODES=$NODES GPUS=$GPUS (total $((NODES*GPUS)) GPUs) IMAGE=$IMAGE CONFIG=$CONFIG"
echo "[submit_sft_q35_image] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME"

# Azure-Blob auto-upload of the final ckpt is enabled by default in the training
# driver. Set AZBLOB_AUTO_PUSH=0 to disable; for pods without workload-identity
# also pass a SAS: AZBLOB_SAS_TOKEN=$(bash scripts/az_ckpt.sh sas | cut -d"'" -f2).
# A SAS contains no commas and the submitter splits each pair on the first '=',
# so it survives --extra-env-vars intact.
EXTRA_ENV="SFT_CONFIG=${CONFIG},NPROC=${GPUS}"
[[ -n "${AZBLOB_AUTO_PUSH:-}" ]] && EXTRA_ENV="${EXTRA_ENV},AZBLOB_AUTO_PUSH=${AZBLOB_AUTO_PUSH}"
[[ -n "${AZBLOB_SAS_TOKEN:-}" ]] && EXTRA_ENV="${EXTRA_ENV},AZBLOB_SAS_TOKEN=${AZBLOB_SAS_TOKEN}"
[[ -n "${AZBLOB_PREFIX:-}" ]]    && EXTRA_ENV="${EXTRA_ENV},AZBLOB_PREFIX=${AZBLOB_PREFIX}"

# Tiny --cmd execs the uploaded driver. SFT_CONFIG + NPROC forwarded via
# --extra-env-vars; HF token/cache come from the echo-rl-creds secret. WAF-safe.
bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" \
    --image "$IMAGE" \
    --node "$NODES" --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --extra-env-vars "$EXTRA_ENV" \
    --follow-logs \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_sft_q35_image.sh'
