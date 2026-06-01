#!/usr/bin/env bash
# In-pod driver for the EVAL-ONLY web-agent job (see docker/submit_eval_easy.sh).
#
# This script is UPLOADED with the repo and executed inside the pod via a tiny
# `--cmd` one-liner. Keeping the heavy setup here (rather than inline in the
# Volcano job YAML) is deliberate: the bonete61 API server sits behind
# Cloudflare, whose WAF blocks `kubectl create` POSTs whose body contains a big
# shell preamble (the `curl ... | sh` installer pattern + assorted URLs). Moving
# that content into an uploaded file keeps the job-create request body tiny and
# WAF-clean.
#
# Required env (injected by submit_job.sh / submit_eval_easy.sh):
#   PVC_MOUNT, USER_ALIAS, JOB_NAME   -- auto-injected by submit_job.sh
#   EVAL_CONFIG                       -- echo_configs/<name>.yaml (path string)
# Secrets (mounted as volumes):
#   /run/secrets/echo-rl-creds/cred.sh        -- BROWSERBASE_*, HF_*, gateway key
#   /run/secrets/echo-rl-openai/OPENAI_API_KEY -- working sk-proj judge key

set -e

echo "[run] hello from $JOB_NAME on $(hostname)"

echo "[run] === source creds ==="
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY
# The phyagi gateway key in cred.sh is dead; use the working sk-proj key from the
# dedicated echo-rl-openai secret and route the OSW judge straight to OpenAI.
if [ -f /run/secrets/echo-rl-openai/OPENAI_API_KEY ]; then
  export OPENAI_API_KEY="$(cat /run/secrets/echo-rl-openai/OPENAI_API_KEY)"
  export OPENAI_GATEWAY_ENDPOINT=''
  echo '[run] OPENAI_API_KEY from echo-rl-openai secret; judge -> api.openai.com'
fi

: "${EVAL_CONFIG:?EVAL_CONFIG must be set (e.g. echo_configs/qwen35_4b_web_agent_easy_eval.yaml)}"

# ------------------------------------------------------------------
# JOB-SCOPED PVC layout (no shared mutable state -> safe to run jobs
# concurrently). Code runs in place from the per-job upload dir.
# ------------------------------------------------------------------
UPLOAD_ROOT=$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME
ENV_ROOT=$PVC_MOUNT/$USER_ALIAS/eval_envs/$JOB_NAME
VENV=$ENV_ROOT/.venv
mkdir -p "$ENV_ROOT/bin"

echo '[run] === install uv (job-scoped) ==='
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ENV_ROOT/bin" INSTALLER_NO_MODIFY_PATH=1 sh
export PATH=$ENV_ROOT/bin:$PATH
uv --version

echo '[run] === create uv venv (system-site-packages inherits torch/vllm/...) ==='
uv venv --system-site-packages --python "$(which python3)" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "[run] python -> $(which python)"
python -V

echo '[run] === uv pip install editables (in place from upload dir) ==='
uv pip install --no-deps \
    -e "$UPLOAD_ROOT/SkyRL" \
    -e "$UPLOAD_ROOT/SkyRL/skyrl-gym" \
    -e "$UPLOAD_ROOT/SkyRL/skyrl-agent" \
    -e "$UPLOAD_ROOT/mini-web-agent"
uv pip install wandb

# ------------------------------------------------------------------
# env paths used by configs
# ------------------------------------------------------------------
export MINI_WEB_AGENT_ROOT=$UPLOAD_ROOT/mini-web-agent
export ECHO_RL_DATA=$MINI_WEB_AGENT_ROOT/data/web_agent
export OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
mkdir -p "$OUTPUT_DIR"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
echo "MINI_WEB_AGENT_ROOT=$MINI_WEB_AGENT_ROOT"
echo "ECHO_RL_DATA=$ECHO_RL_DATA"
echo "OUTPUT_DIR=$OUTPUT_DIR"

echo '[run] === pre-flight ==='
nvidia-smi -L
python -c "import torch, vllm, transformers, wandb, echo_rl, skyrl_gym, skyrl_agent; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__); print('imports OK')"

echo "[run] === launching EVAL-ONLY: $EVAL_CONFIG ==="
# cd into SkyRL so 'python -m echo_rl...' resolves echo_rl to the refreshed
# SkyRL/echo_rl snapshot (cwd is prepended to sys.path by -m) and the
# echo_configs/ path resolves relative to cwd.
cd "$UPLOAD_ROOT/SkyRL"
python -m echo_rl.web_agent.eval_entrypoint --config "$EVAL_CONFIG" || RC=$?
RC=${RC:-0}
echo "[run] eval exited rc=$RC"
echo '[run] === done ==='
