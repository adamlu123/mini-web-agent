#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-echo_configs/qwen3_8b_rl.yaml}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -m echo_rl.terminal_agent.entrypoint --config "$CONFIG_PATH"
