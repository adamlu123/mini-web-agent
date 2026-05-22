#!/usr/bin/env bash
# Batch RL training run — single 8-GPU node, time-capped to 25 min.
# Use this when you want a hands-off "does the real training work?" check.
#
# Cleanup: --cmd uses `timeout 1500 ...` so even if the runner gets wedged,
# the training process self-kills at 25 min and the pod exits.

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
CONFIG="${CONFIG:-echo_configs/qwen35_4b_web_agent_hard_4gpu.yaml}"
TRAIN_TIMEOUT_SEC="${TRAIN_TIMEOUT_SEC:-1500}"   # 25 min hard kill of `python ... entrypoint`

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# Job priority. PRIORITY is just a naming label in JOB_NAME; PRIORITY_CLASS_NAME
# is the actual k8s PriorityClass used by Volcano for scheduling/preemption.
# Override per invocation: PRIORITY=p0 PRIORITY_CLASS_NAME=p0 bash ...
export PRIORITY="${PRIORITY:-high}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"

bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node 8 \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --follow-logs \
    --cmd "set -e
echo '[run] hello from \$JOB_NAME on \$(hostname)'
echo '[run] === source creds (sanitized cred.sh in k8s secret) ==='
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY   # mirrors run_web_agent_yifei.sh

echo '[run] === install editables (~30s) ==='
cd \$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/SkyRL
pip install --no-deps -e . -e ./skyrl-gym -e ./skyrl-agent
cd ../mini-web-agent && pip install --no-deps -e .

echo '[run] === env paths ==='
export MINI_WEB_AGENT_ROOT=\$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/mini-web-agent
export ECHO_RL_DATA=\$MINI_WEB_AGENT_ROOT/data/web_agent
export OUTPUT_DIR=\$PVC_MOUNT/\$USER_ALIAS/outputs/\$JOB_NAME
mkdir -p \$OUTPUT_DIR
export WANDB_MODE=offline
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
echo MINI_WEB_AGENT_ROOT=\$MINI_WEB_AGENT_ROOT
echo ECHO_RL_DATA=\$ECHO_RL_DATA
echo OUTPUT_DIR=\$OUTPUT_DIR

echo '[run] === pre-flight gpu check ==='
nvidia-smi -L

echo '[run] === launching training (timeout=${TRAIN_TIMEOUT_SEC}s) ==='
cd \$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/SkyRL
# timeout returns 124 on cap; we treat that as ok for this validation run
timeout --foreground --signal=TERM --kill-after=30 ${TRAIN_TIMEOUT_SEC} \\
    python -m echo_rl.web_agent.entrypoint --config '${CONFIG}' || RC=\$?
RC=\${RC:-0}
echo \"[run] training exited rc=\$RC (124 = hit time cap, that's ok for smoke)\"
echo '[run] === done ==='
"
