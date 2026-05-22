#!/usr/bin/env bash
# Per-user launcher for the 2-GPU web-agent RL run on Qwen3.5-0.8B over
# om2w *hard* tasks. Mirrors scripts/run_web_agent_hard_4gpu.sh but with
# all paths pointed at /data/t-yifeili/ and ~/cred.sh.
#
# Usage:
#   bash scripts/run_web_agent_yifei.sh
#
# Override knobs (env vars):
#   CONFIG_NAME    — yaml under ${SKYRL_DIR}/echo_configs/ (default 2-GPU 0.8B)
#   OUTPUT_DIR     — checkpoint + log root
#   SKYRL_DIR      — SkyRL repo with hooks applied (default: /data/t-yifeili/SkyRL)
#   ECHO_RL_DATA   — parquet directory (default: ./data/web_agent)
#   CUDA_VISIBLE_DEVICES — defaults to 2,3 since you usually share the box

set -euo pipefail

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SKYRL_DIR:=/data/t-yifeili/SkyRL}"
: "${CONFIG_NAME:=qwen35_0_8b_web_agent_hard_2gpu.yaml}"
: "${OUTPUT_DIR:=${WORKTREE_ROOT}/outputs/qwen35_0_8b_hard_2gpu}"
: "${ECHO_RL_DATA:=${WORKTREE_ROOT}/data/web_agent}"
: "${CUDA_VISIBLE_DEVICES:=2,3}"
export CUDA_VISIBLE_DEVICES

# Personal conda env (built from the same recipe as luyadong's echo-rl env).
source /data/t-yifeili/miniconda3/etc/profile.d/conda.sh
conda activate echo-rl

# Personal credentials. cred.sh sets BROWSERBASE_*, OPENAI_GATEWAY_*,
# OPENAI_API_KEY, HF_TOKEN, HF_HOME, etc. Mirror the upstream launcher
# behaviour of unset'ing OPENAI_GATEWAY_API_KEY so OpenaiEngine falls
# through to the OPENAI_API_KEY path used by the judge.
if [[ -f /home/t-yifeili/cred.sh ]]; then
    # shellcheck disable=SC1091
    source /home/t-yifeili/cred.sh
fi
unset OPENAI_GATEWAY_API_KEY

: "${OPENAI_GATEWAY_ENDPOINT:=http://gateway.phyagi.net/api/responses}"
: "${MINI_WEB_AGENT_ROOT:=/home/luyadong/sandbox/mini-web-agent}"
export OPENAI_GATEWAY_ENDPOINT MINI_WEB_AGENT_ROOT
export OPENAI_API_KEY BROWSERBASE_API_KEY BROWSERBASE_PROJECT_ID HF_TOKEN
export ECHO_RL_DATA OUTPUT_DIR

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p "${OUTPUT_DIR}" "${ECHO_RL_DATA}"

echo "[run_web_agent_yifei] WORKTREE_ROOT=${WORKTREE_ROOT}"
echo "[run_web_agent_yifei] SKYRL_DIR=${SKYRL_DIR}"
echo "[run_web_agent_yifei] CONFIG_NAME=${CONFIG_NAME}"
echo "[run_web_agent_yifei] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run_web_agent_yifei] ECHO_RL_DATA=${ECHO_RL_DATA}"
echo "[run_web_agent_yifei] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_web_agent_yifei] OPENAI_API_KEY set? $([[ -n ${OPENAI_API_KEY-} ]] && echo yes || echo no)"
echo "[run_web_agent_yifei] BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID-<unset>}"

cd "${SKYRL_DIR}"
exec python -m echo_rl.web_agent.entrypoint \
    --config "echo_configs/${CONFIG_NAME}" \
    "$@"
