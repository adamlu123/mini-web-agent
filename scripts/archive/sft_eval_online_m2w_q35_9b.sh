#!/usr/bin/env bash
# Archived: this launcher targets an eval config outside src/miniswewebagent/config.
# Script 2/3 -- ONLINE om2w (Mind2Web-online) eval of the 9B web-agent SFT
# checkpoint. Thin wrapper around run_local_eval.sh: it points the eval driver
# at the trained HF checkpoint (instead of the base Qwen3.5-9B weights) and runs
# a single closed-loop evaluate() pass over the om2w easy train + val parquets,
# scoring each with the OSW judge. This is the REAL agent-success signal
# (script 1 only measures teacher-forced per-turn imitation).
#
# The SFT output of LlamaFactory full finetuning is already HF format
# (safetensors + config + tokenizer), so no FSDP->HF conversion is needed.
#
# Usage:
#   bash scripts/archive/sft_eval_online_m2w_q35_9b.sh
#   CKPT=/path/to/hf/ckpt RUN_TAG=sft9b_ep1 bash scripts/archive/sft_eval_online_m2w_q35_9b.sh
#
# Requires the same online-eval prerequisites as run_local_eval.sh: GPUs +
# Browserbase / judge creds (sourced inside run_local_eval.sh from cred.sh).
set -euo pipefail

REPO=/data/t-yifeili/mini-web-agent
CKPT="${CKPT:-$REPO/LlamaFactory/saves/qwen35_9b/full/websft}"

[[ -d "$CKPT" ]] || { echo "[error] CKPT dir not found: $CKPT"; echo "        train the 9B SFT first, or pass CKPT=/path/to/hf/ckpt"; exit 1; }
ls "$CKPT"/*.safetensors >/dev/null 2>&1 || ls "$CKPT"/model.safetensors.index.json >/dev/null 2>&1 \
  || { echo "[error] $CKPT has no *.safetensors (not an HF-format weights dir)"; exit 1; }

export RUN_TAG="${RUN_TAG:-sft9b}"
echo "[run] online om2w eval of 9B SFT ckpt"
echo "[run] CKPT=$CKPT  RUN_TAG=$RUN_TAG"

# run_local_eval.sh selects configs/qwen35_9b_web_agent_easy_eval.yaml for
# MODEL=9b and overrides trainer.policy.model.path with CKPT.
exec env MODEL=9b CKPT="$CKPT" bash "$REPO/run_local_eval.sh" "$@"
