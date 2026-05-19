#!/usr/bin/env bash
# Launch the web-agent SkyRL training run. Mirrors run_echo_terminal_agent.sh
# but points at the web-agent entrypoint.
set -euo pipefail

: "${OUTPUT_DIR:?Set OUTPUT_DIR=/path/to/outputs/qwen35_4b_web_agent}"
: "${CONFIG_PATH:?Set CONFIG_PATH=configs/qwen35_4b_web_agent.yaml}"
: "${OPENAI_GATEWAY_API_KEY:?Set OPENAI_GATEWAY_API_KEY for the o4-mini judge.}"
: "${OPENAI_GATEWAY_ENDPOINT:=http://gateway.phyagi.net/api/responses}"
export OPENAI_GATEWAY_ENDPOINT

python -m echo_rl.web_agent.entrypoint --config "${CONFIG_PATH}" "$@"
