#!/usr/bin/env bash
# Self-contained launcher for the local 4-GPU web-agent RL run on Qwen3.5-4B
# over om2w *easy* tasks. Mirrors scripts/run_web_agent_hard_4gpu.sh but
# defaults SKYRL_DIR to this user's checkout and points at the easy config.
#
# Usage:
#   bash scripts/run_web_agent_easy_4gpu.sh
#
# Override knobs (env vars):
#   CONFIG_NAME    — yaml under SkyRL/echo_configs/ (default 4B easy 4gpu config)
#   OUTPUT_DIR     — checkpoint + log root (default: ./outputs/qwen35_4b_easy_4gpu)
#   SKYRL_DIR      — SkyRL repo with hooks applied (default: /data/t-yifeili/SkyRL)
#   ECHO_RL_DATA   — parquet directory (default: ./data/web_agent)
#   WANDB_MODE     — set to "disabled" if you don't want to log to wandb

set -euo pipefail

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SKYRL_DIR:=/data/t-yifeili/SkyRL}"
: "${CONFIG_NAME:=qwen35_4b_web_agent_easy_4gpu.yaml}"
: "${OUTPUT_DIR:=${WORKTREE_ROOT}/outputs/qwen35_4b_easy_4gpu}"
: "${ECHO_RL_DATA:=${WORKTREE_ROOT}/data/web_agent}"

# Conda env with SkyRL + echo-rl + flash-attn + vllm + transformers 5.3.
source /home/luyadong/miniconda3/etc/profile.d/conda.sh
conda activate echo-rl

# Credentials — cred.sh sets BROWSERBASE_*, OPENAI_GATEWAY_ENDPOINT,
# HF_TOKEN, and a stale OPENAI_GATEWAY_API_KEY. cred_gateway.sh writes the
# working gateway key into OPENAI_API_KEY. Unset the stale var so the
# OpenaiEngine falls through to the working one.
if [[ -f /home/luyadong/sandbox/cred.sh ]]; then
    # shellcheck disable=SC1091
    source /home/luyadong/sandbox/cred.sh
fi
if [[ -f /home/luyadong/cred_gateway.sh ]]; then
    # shellcheck disable=SC1091
    source /home/luyadong/cred_gateway.sh
fi
unset OPENAI_GATEWAY_API_KEY

: "${OPENAI_GATEWAY_ENDPOINT:=http://gateway.phyagi.net/api/responses}"
: "${MINI_WEB_AGENT_ROOT:=/home/luyadong/sandbox/mini-web-agent}"
export OPENAI_GATEWAY_ENDPOINT MINI_WEB_AGENT_ROOT
export OPENAI_API_KEY BROWSERBASE_API_KEY BROWSERBASE_PROJECT_ID HF_TOKEN
export ECHO_RL_DATA OUTPUT_DIR

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p "${OUTPUT_DIR}" "${ECHO_RL_DATA}"

echo "[run_web_agent_easy_4gpu] WORKTREE_ROOT=${WORKTREE_ROOT}"
echo "[run_web_agent_easy_4gpu] SKYRL_DIR=${SKYRL_DIR}"
echo "[run_web_agent_easy_4gpu] CONFIG_NAME=${CONFIG_NAME}"
echo "[run_web_agent_easy_4gpu] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run_web_agent_easy_4gpu] ECHO_RL_DATA=${ECHO_RL_DATA}"
echo "[run_web_agent_easy_4gpu] OPENAI_API_KEY set? $([[ -n ${OPENAI_API_KEY-} ]] && echo yes || echo no)"
echo "[run_web_agent_easy_4gpu] BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID-<unset>}"
echo "[run_web_agent_easy_4gpu] WANDB_MODE=${WANDB_MODE-<unset>}"

cd "${SKYRL_DIR}"
exec python -m echo_rl.web_agent.entrypoint \
    --config "echo_configs/${CONFIG_NAME}" \
    "$@"
