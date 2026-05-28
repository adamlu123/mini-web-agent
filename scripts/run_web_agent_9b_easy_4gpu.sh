#!/usr/bin/env bash
# Self-contained launcher for the local 4-GPU web-agent RL run on Qwen3.5-9B
# over om2w *easy* tasks. Mirrors scripts/run_web_agent_easy_4gpu.sh but
# points at the 9B model + 4gpu config (tp=1 × 4 engines, gpu_mem_util=0.65).
#
# Usage:
#   bash scripts/run_web_agent_9b_easy_4gpu.sh
#
# Override knobs (env vars):
#   CONFIG_NAME    — yaml under SkyRL/echo_configs/ (default 9B easy 4gpu config)
#   OUTPUT_DIR     — checkpoint + log root (default: ./outputs/qwen35_9b_easy_4gpu)
#   SKYRL_DIR      — SkyRL repo with hooks applied (default: /data/t-yifeili/SkyRL)
#   ECHO_RL_DATA   — parquet directory (default: ./data/web_agent)
#   WANDB_MODE     — set to "disabled" if you don't want to log to wandb

set -euo pipefail

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SKYRL_DIR:=/data/t-yifeili/SkyRL}"
: "${CONFIG_NAME:=qwen35_9b_web_agent_easy_4gpu.yaml}"
: "${OUTPUT_DIR:=${WORKTREE_ROOT}/outputs/qwen35_9b_easy_4gpu}"
: "${ECHO_RL_DATA:=${WORKTREE_ROOT}/data/web_agent}"

# Conda env with SkyRL + echo-rl + vllm 0.19 + torch 2.10+cu128.
# Use yifeili's own miniconda — SkyRL and echo_rl are editable installs from
# /data/t-yifeili/SkyRL and /data/t-yifeili/echo-rl in this env.
source /data/t-yifeili/miniconda3/etc/profile.d/conda.sh
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

# Phyagi gateway and its keys are dead (April 2026); the only working judge
# key on this host is OPENAI_API_BACKUP_KEY (a real sk-proj-...). Overwrite
# OPENAI_API_KEY with it and clear the gateway endpoint so OpenaiEngine
# goes straight to api.openai.com.
if [[ -n "${OPENAI_API_BACKUP_KEY:-}" ]]; then
    export OPENAI_API_KEY="${OPENAI_API_BACKUP_KEY}"
    unset OPENAI_GATEWAY_ENDPOINT
fi

: "${OPENAI_GATEWAY_ENDPOINT:=http://gateway.phyagi.net/api/responses}"
# Point the OSW judge import path at THIS checkout. luyadong's mini-web-agent
# dir on this host has only stale .pyc files (no .py sources) under
# om2w_judge/, so importing om2w_judge.methods.webjudge_online_mind2web from
# there fails and every rollout falls back to failure_reward=0.
: "${MINI_WEB_AGENT_ROOT:=${WORKTREE_ROOT}}"
export OPENAI_GATEWAY_ENDPOINT MINI_WEB_AGENT_ROOT
export OPENAI_API_KEY BROWSERBASE_API_KEY BROWSERBASE_PROJECT_ID HF_TOKEN
export ECHO_RL_DATA OUTPUT_DIR

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p "${OUTPUT_DIR}" "${ECHO_RL_DATA}"

echo "[run_web_agent_9b_easy_4gpu] WORKTREE_ROOT=${WORKTREE_ROOT}"
echo "[run_web_agent_9b_easy_4gpu] SKYRL_DIR=${SKYRL_DIR}"
echo "[run_web_agent_9b_easy_4gpu] CONFIG_NAME=${CONFIG_NAME}"
echo "[run_web_agent_9b_easy_4gpu] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run_web_agent_9b_easy_4gpu] ECHO_RL_DATA=${ECHO_RL_DATA}"
echo "[run_web_agent_9b_easy_4gpu] OPENAI_API_KEY set? $([[ -n ${OPENAI_API_KEY-} ]] && echo yes || echo no)"
echo "[run_web_agent_9b_easy_4gpu] BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID-<unset>}"
echo "[run_web_agent_9b_easy_4gpu] WANDB_MODE=${WANDB_MODE-<unset>}"

cd "${SKYRL_DIR}"
exec python -m echo_rl.web_agent.entrypoint \
    --config "echo_configs/${CONFIG_NAME}" \
    "$@"
