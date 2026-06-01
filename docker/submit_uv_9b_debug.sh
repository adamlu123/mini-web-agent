#!/usr/bin/env bash
# DEBUG variant of submit_uv_9b_easy.sh: identical setup, but on a nonzero
# training exit it dumps the Ray session worker logs + kernel OOM lines to
# stdout (so `kubectl logs` captures the real traceback that otherwise dies
# with the pod in ephemeral /tmp/ray).

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
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"
export PRIORITY="${PRIORITY:-medium}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-medium}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

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
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY
if [ -n \"\${OPENAI_API_KEY_OVERRIDE:-}\" ]; then
  export OPENAI_API_KEY=\"\${OPENAI_API_KEY_OVERRIDE}\"
  export OPENAI_GATEWAY_ENDPOINT=''
fi

CODE_ROOT=\$PVC_MOUNT/\$USER_ALIAS/code
ENV_ROOT=\$PVC_MOUNT/\$USER_ALIAS/envs/echo-rl-uv
VENV=\$ENV_ROOT/.venv
UPLOAD_ROOT=\$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME
mkdir -p \$CODE_ROOT \$ENV_ROOT/bin

if ! command -v rsync >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq rsync
fi
rsync -a --delete --no-perms --no-owner --no-group --no-times \$UPLOAD_ROOT/SkyRL/          \$CODE_ROOT/SkyRL/
rsync -a --delete --no-perms --no-owner --no-group --no-times \$UPLOAD_ROOT/mini-web-agent/ \$CODE_ROOT/mini-web-agent/

if [ ! -x \$ENV_ROOT/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=\$ENV_ROOT/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH=\$ENV_ROOT/bin:\$PATH
if [ ! -d \$VENV ]; then
  uv venv --system-site-packages --python \$(which python3) \$VENV
fi
source \$VENV/bin/activate
uv pip install --no-deps -e \$CODE_ROOT/SkyRL -e \$CODE_ROOT/SkyRL/skyrl-gym -e \$CODE_ROOT/SkyRL/skyrl-agent -e \$CODE_ROOT/mini-web-agent
uv pip install wandb

export MINI_WEB_AGENT_ROOT=\$CODE_ROOT/mini-web-agent
export ECHO_RL_DATA=\$MINI_WEB_AGENT_ROOT/data/web_agent
export OUTPUT_DIR=\$PVC_MOUNT/\$USER_ALIAS/outputs/\$JOB_NAME
mkdir -p \$OUTPUT_DIR
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# keep ray logs and disable dedup so worker tracebacks are complete
export RAY_DEDUP_LOGS=0

echo '[run] === pre-flight ==='
nvidia-smi -L
free -g

echo '[run] === launching training: $CONFIG ==='
cd \$CODE_ROOT/SkyRL
python -m echo_rl.web_agent.entrypoint --config '${CONFIG}' || RC=\$?
RC=\${RC:-0}
echo \"[run] training exited rc=\$RC\"

if [ \"\$RC\" != \"0\" ]; then
  echo '[run] ===================== RAY SESSION LOGS ====================='
  SESS=/tmp/ray/session_latest
  ls -la \$SESS/logs 2>/dev/null | head -60
  for f in \$SESS/logs/worker-*.err \$SESS/logs/worker-*.out \$SESS/logs/python-core-driver-*.log \$SESS/logs/raylet.* \$SESS/logs/dashboard*.log; do
    [ -f \"\$f\" ] || continue
    echo \"##################### \$f #####################\"
    tail -150 \"\$f\" 2>/dev/null
  done
  echo '[run] ===================== KERNEL / OOM ====================='
  dmesg 2>/dev/null | tail -40 || echo '(dmesg unavailable)'
  echo '[run] ===================== copy ray logs to PVC ====================='
  cp -a \$SESS/logs \$OUTPUT_DIR/ray_logs 2>/dev/null && echo \"saved -> \$OUTPUT_DIR/ray_logs\" || echo 'copy failed'
fi
echo '[run] === done ==='
"
