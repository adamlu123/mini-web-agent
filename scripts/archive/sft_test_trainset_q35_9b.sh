#!/usr/bin/env bash
# Archived: this test targets SFT data/configs outside src/miniswewebagent/config.
# Script 1/3 -- OFFLINE sanity test of the 9B web-agent SFT checkpoint on the
# SAME training set it was trained on (web_agent_pae100_full). For every
# assistant turn we teacher-force the preceding context (system + all earlier
# turns) and greedily generate the turn, then check whether the model emits the
# CORRECT web command: i.e. the right action TYPE (<bash> vs <answer>) and, for
# <bash> turns, the exact shell command. This tells you if the model memorised /
# can reproduce the trajectories before you pay for an online rollout (script 2).
#
# This is teacher-forced (each turn conditioned on GOLD history), so it measures
# per-turn imitation, NOT closed-loop agent success -- use script 2 for that.
#
# Usage:
#   bash scripts/archive/sft_test_trainset_q35_9b.sh
#   CKPT=/path/to/hf/ckpt MAX_PROMPT_TOKENS=40000 bash scripts/archive/sft_test_trainset_q35_9b.sh
set -euo pipefail

REPO=/data/t-yifeili/mini-web-agent
PY=/data/t-yifeili/miniconda3/envs/echo-rl/bin/python   # has vllm 0.19 + llamafactory

CKPT="${CKPT:-$REPO/LlamaFactory/saves/qwen35_9b/full/websft}"
DATA="${DATA:-$REPO/LlamaFactory/data/web_agent_pae100_full.json}"
OUT="${OUT:-$REPO/eval_outputs/sft9b_trainset_$(date +%Y%m%d_%H%M%S)}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-40000}"   # skip turns whose gold prefix exceeds this
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

[[ -d "$CKPT" ]] || { echo "[error] CKPT dir not found: $CKPT"; echo "        train the 9B SFT first, or pass CKPT=/path/to/hf/ckpt"; exit 1; }
ls "$CKPT"/*.safetensors >/dev/null 2>&1 || ls "$CKPT"/model.safetensors.index.json >/dev/null 2>&1 \
  || { echo "[error] $CKPT has no *.safetensors (not an HF-format weights dir)"; exit 1; }
[[ -f "$DATA" ]] || { echo "[error] DATA not found: $DATA"; exit 1; }
mkdir -p "$OUT"

echo "[run] CKPT=$CKPT"
echo "[run] DATA=$DATA"
echo "[run] OUT=$OUT  MAX_PROMPT_TOKENS=$MAX_PROMPT_TOKENS"
nvidia-smi -L || true

# Run a REAL .py module (not a stdin heredoc): vLLM spawns worker subprocesses
# that re-import __main__, which fails with FileNotFoundError <stdin> for a
# `python -` script. Env vars carry the config.
CKPT="$CKPT" DATA="$DATA" OUT="$OUT" \
MAX_PROMPT_TOKENS="$MAX_PROMPT_TOKENS" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
TP="${TP:-1}" \
"$PY" "$REPO/scripts/sft_test_trainset.py"
echo "[run] done -> $OUT/summary.json"
