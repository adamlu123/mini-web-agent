#!/usr/bin/env bash
# Launch a PURE-IMAGE training job on a single 8xB200 node, runs to completion.
# Same environment as submit_debug_pure_image.sh -- it uses ONLY the container
# image's Python (the persistent uv venv on the PVC is intentionally NOT used);
# the four editables are pip-installed straight into the image Python by the
# uploaded driver (docker/run_train_pure_image.sh). The only difference vs the
# debug variant: the driver launches training (python -m
# echo_rl.web_agent.entrypoint) instead of sleeping.
#
# The actual setup + training launch lives in the uploaded driver, and `--cmd`
# is just a tiny exec of it -- this keeps the `kubectl create` request body small
# and clear of the Cloudflare WAF that blocks big inline shell preambles.
#
#   PRIORITY=p0 PRIORITY_CLASS_NAME=high bash docker/submit_train_pure_image.sh
#   CONFIG=echo_configs/qwen35_9b_web_agent_hard_8gpu.yaml bash docker/submit_train_pure_image.sh
#
# Kill manually with:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
GPUS="${GPUS:-8}"
# Config path is relative to SkyRL/ (echo_configs lives inside the uploaded SkyRL).
CONFIG="${CONFIG:-echo_configs/qwen35_4b_web_agent_hard_4gpu.yaml}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# WandB host. submit_job.sh defaults WANDB_HOST to microsoft-research.wandb.io,
# which 401s the personal public-wandb WANDB_API_KEY. Force the public host so the
# host-shell WANDB_API_KEY routes to https://api.wandb.ai. Override at call
# time if you want MS-internal: WANDB_HOST=https://microsoft-research.wandb.io ...
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"

# Job priority. PRIORITY is just a naming label in JOB_NAME (the GPU monitor
# dashboard buckets on it -> p0). PRIORITY_CLASS_NAME is the REAL k8s
# PriorityClass and MUST be one of high/medium/low -- 'p0' is NOT a valid class
# and would fail admission, so p0 maps to the 'high' class here.
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
# Submit under the 'cua' workstream by default (workstream label = PROJECT_NAME).
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_train_pure_image] GPUS=$GPUS IMAGE=$IMAGE CONFIG=$CONFIG"
echo "[submit_train_pure_image] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME WANDB_HOST=$WANDB_HOST"

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
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_train_pure_image.sh'
