#!/usr/bin/env bash
# DEBUG variant of submit_real_train_batch.sh: identical local-install path,
# but (a) tees ALL training stdout/stderr to $OUTPUT_DIR/full_stdout.log on the
# PVC (survives the vcjob PodFailed->AbortJob force-delete), (b) runs a
# background GPU/host-memory logger to $OUTPUT_DIR/gpu_mem.log every 5s so an
# OOM is visible, (c) on exit copies the ray session logs to the PVC.

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
CONFIG="${CONFIG:-echo_configs/qwen35_9b_web_agent_easy_8gpu.yaml}"

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
    --node 1 --gpu-per-node 8 \
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

echo '[run] === install editables (~30s) ==='
cd \$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/SkyRL
pip install --no-deps -e . -e ./skyrl-gym -e ./skyrl-agent
cd ../mini-web-agent && pip install --no-deps -e .
pip install wandb

export MINI_WEB_AGENT_ROOT=\$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/mini-web-agent
export ECHO_RL_DATA=\$MINI_WEB_AGENT_ROOT/data/web_agent
export OUTPUT_DIR=\$PVC_MOUNT/\$USER_ALIAS/outputs/\$JOB_NAME
mkdir -p \$OUTPUT_DIR
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
echo OUTPUT_DIR=\$OUTPUT_DIR

echo '[run] === pre-flight gpu check ==='
nvidia-smi -L

# background GPU + host-mem sampler -> PVC (so an OOM is visible post-mortem)
( while true; do
    echo \"=== \$(date -u +%H:%M:%S) ===\";
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader;
    free -g | sed -n '1,2p';
  done > \$OUTPUT_DIR/gpu_mem.log 2>&1 ) &
SAMPLER=\$!

echo '[run] === launching training (tee -> PVC) ==='
cd \$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/SkyRL
RC=0
stdbuf -oL -eL python -u -m echo_rl.web_agent.entrypoint --config '${CONFIG}' 2>&1 \
  | stdbuf -oL tee \$OUTPUT_DIR/full_stdout.log
RC=\${PIPESTATUS[0]}
kill \$SAMPLER 2>/dev/null || true
echo \"[run] training exited rc=\$RC\"

echo '[run] === post-mortem -> PVC ==='
{
  echo \"#### final nvidia-smi ####\"; nvidia-smi || true;
  echo \"#### dmesg tail (OOM?) ####\"; dmesg 2>/dev/null | tail -50 || echo '(dmesg unavailable)';
} > \$OUTPUT_DIR/postmortem.log 2>&1 || true
cp -a /tmp/ray/session_latest/logs \$OUTPUT_DIR/ray_logs 2>/dev/null || true
echo '[run] === done ==='
"
