#!/usr/bin/env bash
# Submit RL training to bonete61 using a persistent uv venv on Vast PVC.
#
# First run:
#   - installs uv to $ENV_ROOT/bin (PVC, persists across jobs)
#   - creates .venv with --system-site-packages so it inherits the image's
#     torch / vllm / flash-attn / transformers / ... (no expensive reinstall)
#   - rsyncs uploaded code (SkyRL + mini-web-agent) to a stable PVC location
#   - uv pip install --no-deps -e [skyrl, skyrl-gym, skyrl-agent, mini-web-agent]
#   - uv pip install wandb (the one dep the image is missing)
# Subsequent runs: same steps; everything except the rsync is a near-noop.
#
# Override config / image via env:
#   CONFIG=echo_configs/qwen35_9b_web_agent_hard_8gpu.yaml bash docker/submit_uv_9b_easy.sh
#
# Persistent PVC layout (after first successful run):
#   /mnt/pvc/<alias>/code/SkyRL/           — stable, rsynced each job
#   /mnt/pvc/<alias>/code/mini-web-agent/  — stable, rsynced each job
#   /mnt/pvc/<alias>/envs/echo-rl-uv/.venv — uv venv, .pth points at code/
#   /mnt/pvc/<alias>/envs/echo-rl-uv/bin/uv — uv binary

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
CONFIG="${CONFIG:-echo_configs/qwen35_9b_web_agent_easy_8gpu.yaml}"
GPUS="${GPUS:-8}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# Public WandB by default (matches submit_real_train_batch.sh)
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"

# Naming for the GPU monitor dashboard
export PRIORITY="${PRIORITY:-medium}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-medium}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

# Forward local OPENAI_API_BACKUP_KEY into the pod as OPENAI_API_KEY_OVERRIDE,
# which the --cmd preamble below picks up to route the om2w judge through
# api.openai.com instead of the phyagi gateway.
EXTRA_ENV_ARGS=()
if [[ -n "${OPENAI_API_BACKUP_KEY:-}" ]]; then
    EXTRA_ENV_ARGS+=(--extra-env-vars "OPENAI_API_KEY_OVERRIDE=${OPENAI_API_BACKUP_KEY}")
fi

bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    "${EXTRA_ENV_ARGS[@]}" \
    --follow-logs \
    --cmd "set -e
echo '[run] hello from \$JOB_NAME on \$(hostname)'

echo '[run] === source creds ==='
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY
if [ -n \"\${OPENAI_API_KEY_OVERRIDE:-}\" ]; then
  export OPENAI_API_KEY=\"\${OPENAI_API_KEY_OVERRIDE}\"
  export OPENAI_GATEWAY_ENDPOINT=''
  echo '[run] OPENAI_API_KEY overridden; judge -> api.openai.com'
fi

# ------------------------------------------------------------------
# stable PVC layout
# ------------------------------------------------------------------
CODE_ROOT=\$PVC_MOUNT/\$USER_ALIAS/code
ENV_ROOT=\$PVC_MOUNT/\$USER_ALIAS/envs/echo-rl-uv
VENV=\$ENV_ROOT/.venv
UPLOAD_ROOT=\$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME
mkdir -p \$CODE_ROOT \$ENV_ROOT/bin

echo '[run] === copy uploaded code to stable PVC path ==='
# Image lacks rsync; cp -a fails on NFS perms-preservation. Use rsync after
# apt-installing it (idempotent, ~10s first time, cache hit later).
if ! command -v rsync >/dev/null 2>&1; then
  echo '[run] installing rsync (one-time per pod)...'
  apt-get update -qq && apt-get install -y -qq rsync
fi
rsync -a --delete --no-perms --no-owner --no-group --no-times \
    \$UPLOAD_ROOT/SkyRL/          \$CODE_ROOT/SkyRL/
rsync -a --delete --no-perms --no-owner --no-group --no-times \
    \$UPLOAD_ROOT/mini-web-agent/ \$CODE_ROOT/mini-web-agent/

# ------------------------------------------------------------------
# uv install (idempotent — only downloads if missing)
# ------------------------------------------------------------------
if [ ! -x \$ENV_ROOT/bin/uv ]; then
  echo '[run] === install uv to \$ENV_ROOT/bin ==='
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=\$ENV_ROOT/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH=\$ENV_ROOT/bin:\$PATH
uv --version

# ------------------------------------------------------------------
# venv (system-site-packages inherits torch / vllm / flash-attn / ...)
# ------------------------------------------------------------------
if [ ! -d \$VENV ]; then
  echo '[run] === create uv venv on PVC ==='
  uv venv --system-site-packages --python \$(which python3) \$VENV
fi
# shellcheck disable=SC1091
source \$VENV/bin/activate
echo \"[run] python -> \$(which python)\"
python -V

# ------------------------------------------------------------------
# editable installs (writes .pth files pointing at stable PVC code paths;
# survives across jobs, so subsequent submits just re-rsync and go)
# ------------------------------------------------------------------
echo '[run] === uv pip install editables (idempotent) ==='
uv pip install --no-deps \
    -e \$CODE_ROOT/SkyRL \
    -e \$CODE_ROOT/SkyRL/skyrl-gym \
    -e \$CODE_ROOT/SkyRL/skyrl-agent \
    -e \$CODE_ROOT/mini-web-agent
# image is missing wandb; install once into the PVC venv
uv pip install wandb

# ------------------------------------------------------------------
# env paths used by configs
# ------------------------------------------------------------------
export MINI_WEB_AGENT_ROOT=\$CODE_ROOT/mini-web-agent
export ECHO_RL_DATA=\$MINI_WEB_AGENT_ROOT/data/web_agent
export OUTPUT_DIR=\$PVC_MOUNT/\$USER_ALIAS/outputs/\$JOB_NAME
mkdir -p \$OUTPUT_DIR
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
echo MINI_WEB_AGENT_ROOT=\$MINI_WEB_AGENT_ROOT
echo ECHO_RL_DATA=\$ECHO_RL_DATA
echo OUTPUT_DIR=\$OUTPUT_DIR

echo '[run] === pre-flight ==='
nvidia-smi -L
python -c \"import torch, vllm, transformers, wandb, echo_rl, skyrl_gym, skyrl_agent; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__); print('transformers', transformers.__version__); print('imports OK')\"

echo '[run] === launching training: $CONFIG ==='
cd \$CODE_ROOT/SkyRL
python -m echo_rl.web_agent.entrypoint --config '${CONFIG}' || RC=\$?
RC=\${RC:-0}
echo \"[run] training exited rc=\$RC\"
echo '[run] === done ==='
"
