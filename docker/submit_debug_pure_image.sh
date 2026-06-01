#!/usr/bin/env bash
# Launch a long-lived 8xB200 DEBUG pod that uses ONLY the container image's
# Python environment -- the persistent uv venv on the PVC is intentionally NOT
# used. The four editables are pip-installed straight into the image Python by
# the uploaded driver (docker/run_debug_pure_image.sh). API creds are still
# synced in (via the echo-rl-creds + echo-rl-openai k8s secrets).
#
# Submitted DETACHED (no blocking TTY) so it can be driven by tooling; the pod
# runs `sleep infinity` after bootstrap. The actual setup lives in the uploaded
# driver, and `--cmd` is just a tiny exec of it -- this keeps the `kubectl
# create` request body small and clear of the Cloudflare WAF that blocks big
# inline shell preambles.
#
#   bash docker/submit_debug_pure_image.sh
#   GPUS=8 bash docker/submit_debug_pure_image.sh   # override gpu count
#
# After it returns, exec into the pod (wait for the .debug_ready marker):
#   kubectl -n bonete61 exec -it <pod> -- bash
#
# Tear down when done:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
GPUS="${GPUS:-8}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# WandB host. submit_job.sh defaults WANDB_HOST to microsoft-research.wandb.io,
# which 401s our personal public-wandb WANDB_API_KEY. Force the public host so the
# host-shell WANDB_API_KEY routes to https://api.wandb.ai. Override at call
# time if you want MS-internal: WANDB_HOST=https://microsoft-research.wandb.io ...
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"

# Naming for the GPU monitor dashboard (see submit_real_train_batch.sh).
export PRIORITY="${PRIORITY:-medium}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-medium}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_debug_pure_image] GPUS=$GPUS IMAGE=$IMAGE (no uv venv; image Python only)"

# Tiny --cmd: just exec the uploaded driver. Creds come from the two secret
# volumes -- no raw secret in the request body, so WAF-safe.
bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --secret-volume echo-rl-openai:/run/secrets/echo-rl-openai \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_debug_pure_image.sh'
