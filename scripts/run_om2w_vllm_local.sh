#!/bin/bash
# Run the om2w benchmark against a locally served Qwen3.5 checkpoint.
# Start the server first with scripts/serve_vllm_qwen35.sh.
#
#   CFG=eval/om2w_spb_vllm_sw10.yaml OUT=outputs/qwen35_4b_sw10 \
#     bash scripts/run_om2w_vllm.sh
#
# Defaults to all 300 tasks; set LIMIT/TASK_LEVEL to subset.
set -euo pipefail
source /home/luyadong/cred.sh

REPO="${REPO:-/home/luyadong/sandbox/mini-web-agent}"
VENV_BIN="${VENV_BIN:-/home/luyadong/.venv/bin}"
CFG="${CFG:-eval/om2w_spb_vllm_sw10.yaml}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
TASKS_FILE="${TASKS_FILE:-$REPO/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
WORKERS="${WORKERS:-8}"
OUT="${OUT:-$REPO/outputs/qwen35_vllm}"

# The agent shells out to `python -m browser_session`; hosts that only ship
# `python3` need the venv bin first on PATH or every task fails with
# "python: command not found".
export PATH="$VENV_BIN:$PATH"

cd "$REPO"
mkdir -p "$OUT"

extra=()
[ -n "${LIMIT:-}" ] && extra+=(--limit "$LIMIT")
[ -n "${TASK_LEVEL:-}" ] && extra+=(--task-level "$TASK_LEVEL")

exec "$VENV_BIN/python" -m miniswewebagent.run.benchmarks.om2w \
  -c "$CFG" \
  -c "model.endpoint=$ENDPOINT" \
  -c "model.model_name=$MODEL_NAME" \
  --tasks-file "$TASKS_FILE" \
  --workers "$WORKERS" \
  --output-dir "$OUT" "${extra[@]}" 2>&1 | tee -a "$OUT/run.log"
