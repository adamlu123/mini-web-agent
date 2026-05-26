#!/usr/bin/env bash
# Batch RL training run — single 8-GPU node, runs until completion.
# Kill manually with `kubectl -n bonete61 delete job.batch.volcano.sh/<job>`
# (no in-script time cap).

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
CONFIG="${CONFIG:-echo_configs/qwen35_4b_web_agent_hard_4gpu.yaml}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# WandB host. submit_job.sh defaults WANDB_HOST to microsoft-research.wandb.io,
# which 401s our public-wandb WANDB_API_KEY. Force the public host so the
# host-shell WANDB_API_KEY routes to https://api.wandb.ai. Override at call
# time if you want MS-internal: WANDB_HOST=https://microsoft-research.wandb.io ...
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"

# Job priority. PRIORITY is just a naming label in JOB_NAME; PRIORITY_CLASS_NAME
# is the actual k8s PriorityClass used by Volcano for scheduling/preemption.
# Override per invocation: PRIORITY=p0 PRIORITY_CLASS_NAME=p0 bash ...
export PRIORITY="${PRIORITY:-medium}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-medium}"

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

# Optional fallback: when OPENAI_API_KEY_OVERRIDE is set in the host shell
# at submission time, swap in that key and clear the phyagi gateway endpoint
# so the o4-mini judge calls api.openai.com directly. Used while the
# msr/phyagi credentials are dead. Leave OPENAI_API_KEY_OVERRIDE unset to
# keep the old cred.sh-driven behaviour.
if [ -n '${OPENAI_API_KEY_OVERRIDE:-}' ]; then
  export OPENAI_API_KEY='${OPENAI_API_KEY_OVERRIDE:-}'
  export OPENAI_GATEWAY_ENDPOINT=''
  echo '[run] OPENAI_API_KEY overridden; judge -> api.openai.com'
fi

echo '[run] === install editables (~30s) ==='
cd \$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/SkyRL
pip install --no-deps -e . -e ./skyrl-gym -e ./skyrl-agent
cd ../mini-web-agent && pip install --no-deps -e .
# Base image lacks wandb; install it so logger=wandb works.
pip install wandb

echo '[run] === env paths ==='
export MINI_WEB_AGENT_ROOT=\$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/mini-web-agent
export ECHO_RL_DATA=\$MINI_WEB_AGENT_ROOT/data/web_agent
export OUTPUT_DIR=\$PVC_MOUNT/\$USER_ALIAS/outputs/\$JOB_NAME
mkdir -p \$OUTPUT_DIR
# WandB: submit_job.sh injects API_KEY / BASE_URL / HOST / PROJECT / NAME.
# Leave WANDB_MODE unset to enable online sync to microsoft-research.wandb.io.
# (Re-export WANDB_MODE=offline only if the cluster blocks egress to wandb.)
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
echo MINI_WEB_AGENT_ROOT=\$MINI_WEB_AGENT_ROOT
echo ECHO_RL_DATA=\$ECHO_RL_DATA
echo OUTPUT_DIR=\$OUTPUT_DIR

echo '[run] === pre-flight gpu check ==='
nvidia-smi -L

echo '[run] === launching training (no time cap) ==='
cd \$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/SkyRL
python -m echo_rl.web_agent.entrypoint --config '${CONFIG}' || RC=\$?
RC=\${RC:-0}
echo \"[run] training exited rc=\$RC\"
echo '[run] === done ==='
"
