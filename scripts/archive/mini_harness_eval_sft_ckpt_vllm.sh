#!/usr/bin/env bash
# Archived: its default benchmark YAML is no longer in src/miniswewebagent/config.
# 用历史 miniswewebagent harness 评测 SFT 后的 HuggingFace 格式 checkpoint。
# 这个 wrapper 固定使用 SFT prompt 与 <think>/<bash>/<answer> 解析配置；
# 其余运行参数仍由 scripts/mini_harness_eval_sft_vllm.sh 统一处理。

set -euo pipefail

REPO="${REPO:-/data/t-yifeili/mini-web-agent}"
CKPT="${CKPT:-/data/t-yifeili/ckpts/websft_32k}"
if [[ -z "${MODEL_NAME:-}" ]]; then
	MODEL_NAME="$(basename "$CKPT")"
fi
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/om2w_sft_vllm.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/sft_ckpt_vllm/${MODEL_NAME}_${TASK_LEVEL:-easy}_$(date +%Y%m%d_%H%M%S)}"

export REPO CKPT MODEL_NAME BENCHMARK_CONFIG OUTPUT_DIR

exec bash "$REPO/scripts/mini_harness_eval_sft_vllm.sh" "$@"
