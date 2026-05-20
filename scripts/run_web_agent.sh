#!/usr/bin/env bash
# Launch the web-agent SkyRL training run. Mirrors run_echo_terminal_agent.sh
# but points at the web-agent entrypoint and threads the judge / browserbase
# credentials through.
set -euo pipefail

: "${OUTPUT_DIR:?Set OUTPUT_DIR=/path/to/outputs/qwen35_4b_web_agent}"
: "${CONFIG_PATH:?Set CONFIG_PATH=configs/qwen35_4b_web_agent_hard_4gpu.yaml}"

# Source local cred files if present so OPENAI_API_KEY (gateway), browserbase
# keys, and the gateway endpoint are exported into the training process. The
# stale OPENAI_GATEWAY_API_KEY in cred.sh is unset so OpenaiEngine falls
# through to the working OPENAI_API_KEY from cred_gateway.sh.
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
export ECHO_RL_DATA="${ECHO_RL_DATA:-$(pwd)/data/web_agent}"
mkdir -p "${OUTPUT_DIR}" "${ECHO_RL_DATA}"

echo "[run_web_agent] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run_web_agent] CONFIG_PATH=${CONFIG_PATH}"
echo "[run_web_agent] ECHO_RL_DATA=${ECHO_RL_DATA}"
echo "[run_web_agent] MINI_WEB_AGENT_ROOT=${MINI_WEB_AGENT_ROOT}"
echo "[run_web_agent] OPENAI_API_KEY set? $([[ -n ${OPENAI_API_KEY-} ]] && echo yes || echo no)"
echo "[run_web_agent] BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID-<unset>}"

python -m echo_rl.web_agent.entrypoint --config "${CONFIG_PATH}" "$@"
