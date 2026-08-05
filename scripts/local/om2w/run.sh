#!/usr/bin/env bash
# Run the om2w benchmark against a locally served Qwen3.5 checkpoint.
# Start the server first with scripts/local/om2w/serve_qwen35.sh.
#
#   CFG=eval/om2w_spb_vllm_sw10.yaml OUT=outputs/qwen35_4b_r1 bash scripts/local/om2w/run.sh
#
# Defaults to all 300 tasks; set LIMIT/TASK_LEVEL to subset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VENV_ROOT="${VIRTUAL_ENV:-${HOME}/.venv}"
VENV_BIN="${VENV_BIN:-$VENV_ROOT/bin}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-${CRED_FILE:-${HOME}/cred.sh}}"
[[ -f "$CREDENTIALS_FILE" ]] || {
  echo "credentials file not found: $CREDENTIALS_FILE" >&2
  exit 1
}
[[ -x "$VENV_BIN/python" ]] || {
  echo "Python executable not found: $VENV_BIN/python" >&2
  exit 1
}
# shellcheck disable=SC1090
source "$CREDENTIALS_FILE"
CFG="${CFG:-eval/om2w_spb_vllm_sw10.yaml}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8002/v1}"
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

# Extra dotted config overrides, space separated, e.g.
#   EXTRA_CFG="agent.step_limit=10 run.judge_runs=3"
# The policy port does NOT need one: ENDPOINT sets model.endpoint, which is the
# single source of truth the agent mirrors into the workspace.
cfg_overrides=()
for override in ${EXTRA_CFG:-}; do
  cfg_overrides+=(-c "$override")
done

exec "$VENV_BIN/python" -m miniswewebagent.run.benchmarks.om2w \
  -c "$CFG" \
  -c "model.endpoint=$ENDPOINT" \
  -c "model.model_name=$MODEL_NAME" \
  "${cfg_overrides[@]}" \
  --tasks-file "$TASKS_FILE" \
  --workers "$WORKERS" \
  --output-dir "$OUT" "${extra[@]}" 2>&1 | tee -a "$OUT/run.log"
