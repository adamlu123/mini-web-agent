#!/usr/bin/env bash
# In-pod driver for a PURE-IMAGE training run (no sleep -- runs to completion).
#
# Same environment as run_debug_pure_image.sh: it DISCARDS uv entirely and uses
# the container image's own Python (the echo-rl image already bakes the 269
# runtime deps), `pip install --no-deps -e` the four editables (SkyRL, skyrl-gym,
# skyrl-agent, mini-web-agent) straight into the system interpreter. The ONLY
# difference vs the debug driver: instead of writing an activate script and
# `sleep infinity`, this one launches training and exits with the trainer's rc.
#
# UPLOADED with the repo and executed inside the pod via a tiny `--cmd` one-liner
# (see docker/submit_train_pure_image.sh). Keeping the heavy setup here -- not
# inline in the Volcano job YAML -- keeps the `kubectl create` POST body clear of
# the bonete61 Cloudflare WAF that blocks big inline shell preambles.
#
# Required env (auto-injected by submit_job.sh):
#   PVC_MOUNT, USER_ALIAS, JOB_NAME
# Required env (forwarded by submit_train_pure_image.sh via --extra-env-vars):
#   TRAIN_CONFIG   -- config path relative to SkyRL/ (e.g. echo_configs/...yaml)
# Secrets (mounted as volumes):
#   /run/secrets/echo-rl-creds/cred.sh         -- BROWSERBASE_*, HF_*, gateway key
#   /run/secrets/echo-rl-openai/OPENAI_API_KEY -- working sk-proj judge key

set -e

echo "[boot] pure-image TRAIN pod $JOB_NAME on $(hostname)"
echo "[boot] TRAIN_CONFIG=${TRAIN_CONFIG:?TRAIN_CONFIG not set}"

CODE_ROOT=$PVC_MOUNT/$USER_ALIAS/code
UPLOAD_ROOT=$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME
OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
mkdir -p "$CODE_ROOT" "$OUTPUT_DIR"

echo '[boot] === copy uploaded code to stable PVC path ==='
if ! command -v rsync >/dev/null 2>&1; then
  echo '[boot] installing rsync (one-time per pod)...'
  apt-get update -qq && apt-get install -y -qq rsync
fi
rsync -a --delete --no-perms --no-owner --no-group --no-times \
    "$UPLOAD_ROOT/SkyRL/"          "$CODE_ROOT/SkyRL/"
rsync -a --delete --no-perms --no-owner --no-group --no-times \
    "$UPLOAD_ROOT/mini-web-agent/" "$CODE_ROOT/mini-web-agent/"

echo '[boot] === pip install editables into the IMAGE Python (no venv, no uv) ==='
echo "[boot] python -> $(command -v python) ; $(python -V 2>&1)"
pip install --no-deps --no-build-isolation \
    -e "$CODE_ROOT/SkyRL" \
    -e "$CODE_ROOT/SkyRL/skyrl-gym" \
    -e "$CODE_ROOT/SkyRL/skyrl-agent" \
    -e "$CODE_ROOT/mini-web-agent"
python -c "import wandb" 2>/dev/null || pip install wandb

echo '[boot] === source creds + route OSW judge to api.openai.com ==='
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY
# The phyagi gateway key in cred.sh is dead; use the working sk-proj key from the
# echo-rl-openai secret and route the OSW judge straight to api.openai.com.
if [ -f /run/secrets/echo-rl-openai/OPENAI_API_KEY ]; then
  export OPENAI_API_KEY="$(cat /run/secrets/echo-rl-openai/OPENAI_API_KEY)"
  export OPENAI_GATEWAY_ENDPOINT=''
  echo '[boot] OPENAI_API_KEY set from echo-rl-openai secret; judge -> api.openai.com'
fi

echo '[boot] === env paths ==='
export MINI_WEB_AGENT_ROOT=$CODE_ROOT/mini-web-agent
export ECHO_RL_DATA=$CODE_ROOT/mini-web-agent/data/web_agent
export OUTPUT_DIR=$OUTPUT_DIR
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# WandB: submit_job.sh injects API_KEY / BASE_URL / HOST / PROJECT / NAME.
# Leave WANDB_MODE unset so curves sync online (host forced to api.wandb.ai by
# the submit script for the personal public-wandb key).
echo "[boot] MINI_WEB_AGENT_ROOT=$MINI_WEB_AGENT_ROOT"
echo "[boot] ECHO_RL_DATA=$ECHO_RL_DATA"
echo "[boot] OUTPUT_DIR=$OUTPUT_DIR"

echo '[boot] === pre-flight ==='
nvidia-smi -L
python -c "import torch, vllm, transformers, wandb, echo_rl, skyrl_gym, skyrl_agent; print('torch', torch.__version__); print('vllm', vllm.__version__); print('imports OK')"

echo '[boot] === launching training (no time cap) ==='
cd "$CODE_ROOT/SkyRL"
RC=0
python -m echo_rl.web_agent.entrypoint --config "$TRAIN_CONFIG" || RC=$?
echo "[boot] training exited rc=$RC"
echo '[boot] === done ==='
exit "$RC"
