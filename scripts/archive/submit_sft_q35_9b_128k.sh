#!/usr/bin/env bash
# Archived: this submitter targets a LlamaFactory YAML outside src/miniswewebagent/config.
# Script 3/3 -- launch a second 9B web-agent SFT run with cutoff_len = 128k
# tokens (qwen35_9b_websft_128k.yaml) so every untruncated convo (max ~89k tok)
# is kept whole. Thin wrapper around the existing cluster submitter
# docker/submit_sft_q35_image_debug.sh (8xB200 single node).
#
# !!! See the MEMORY WARNING at the top of the 128k yaml: 128k x ~248k-vocab
# logits is huge and may OOM on a 180GB B200 without a fused/chunked CE. !!!
#
# Usage:
#   bash scripts/archive/submit_sft_q35_9b_128k.sh
#   WANDB_PROJECT=qwen35_9b_websft_128k bash scripts/archive/submit_sft_q35_9b_128k.sh
set -euo pipefail

REPO="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
CONFIG="examples/train_full/qwen35_9b_websft_128k.yaml"

[[ -f "$REPO/LlamaFactory/$CONFIG" ]] || { echo "[error] config not found: LlamaFactory/$CONFIG"; exit 1; }

export WANDB_PROJECT="${WANDB_PROJECT:-qwen35_9b_websft_128k}"
echo "[submit] 9B SFT @128k -> CONFIG=$CONFIG  WANDB_PROJECT=$WANDB_PROJECT"

exec env CONFIG="$CONFIG" bash "$REPO/docker/submit_sft_q35_image_debug.sh" "$@"
