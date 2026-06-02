#!/usr/bin/env bash
# Launch a long-lived 8xB200 DEBUG pod on the *generic* qwen3.5 image
#   aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415
# instead of our own t-yifeili/echo-rl image. The point: qwen3.5 is reported to
# work on this image's newer vllm 0.18.0 / torch 2.10 / transformer-engine 2.13
# stack (vs. the weight-corruption gibberish seen on the echo-rl image), so this
# pod lets us test echo_rl.web_agent against that known-good runtime.
#
# Unlike submit_debug_pure_image.sh, this image does NOT bake docker/
# requirements.txt; the uploaded driver (docker/run_debug_q35_image.sh) installs
# only the deps the image is MISSING (keeping its baked vllm/torch/te stack
# intact) and then `--no-deps -e`'s the four editables.
#
# Submitted DETACHED (no blocking TTY); the pod runs `sleep infinity` after
# bootstrap. The heavy setup lives in the uploaded driver and `--cmd` is just a
# tiny exec of it -- keeps the `kubectl create` request body clear of the
# Cloudflare WAF that blocks big inline shell preambles.
#
#   bash docker/submit_debug_q35_image.sh
#   GPUS=8 bash docker/submit_debug_q35_image.sh   # override gpu count
#
# After it returns, exec into the pod (wait for the .debug_ready marker under
# $PVC/envs/q35-image/):
#   kubectl -n bonete61 exec -it <pod> -- bash
#
# Tear down when done:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
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
# p0/high so the debug pod isn't preempted/evicted under cluster pressure.
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_debug_q35_image] GPUS=$GPUS IMAGE=$IMAGE (generic qwen3.5 image; install only missing deps)"

# Tiny --cmd: just exec the uploaded driver. Creds come from the two secret
# volumes -- no raw secret in the request body, so WAF-safe.
bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --secret-volume echo-rl-openai:/run/secrets/echo-rl-openai \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_debug_q35_image.sh'
