#!/usr/bin/env bash
# Interactive 8×B200 pod with echo-rl image + creds secret mounted.
# Drops you into a bash shell with full env ready to run RL training.
#
# One-time setup (already done):
#   kubectl create secret generic echo-rl-creds \
#       --from-file=cred.sh=<path to sanitized cred> -n bonete61
#
# Usage:
#   bash docker/submit_interactive_8gpu.sh
#
# Inside the pod:
#   source /run/secrets/echo-rl-creds/cred.sh
#   unset OPENAI_GATEWAY_API_KEY    # mirrors run_web_agent_yifei.sh
#   cd $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/SkyRL
#   pip install --no-deps -e . -e ./skyrl-gym -e ./skyrl-agent
#   cd ../mini-web-agent && pip install --no-deps -e .
#   export ECHO_RL_DATA=$(pwd)/data/web_agent
#   export OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
#   mkdir -p $OUTPUT_DIR $ECHO_RL_DATA
#   cd ../SkyRL
#   python -m echo_rl.web_agent.entrypoint --config echo_configs/qwen35_4b_web_agent_hard_4gpu.yaml
#
# Ctrl-D when done → job auto-deleted, GPUs released.

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# Job priority. See submit_real_train_batch.sh for explanation.
export PRIORITY="${PRIORITY:-high}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"

bash "$SUBMIT" \
    --interactive \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node 8 \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds
