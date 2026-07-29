#!/usr/bin/env bash
# Serve Qwen3.5 through vLLM and run the mini-web-agent OM2W inference harness.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-qwen35_9b_base}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-http://${HOST}:${PORT}/v1/chat/completions}"
TASKS_FILE="${TASKS_FILE:-$REPO/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
TASK_LEVEL="${TASK_LEVEL:-easy}"
LIMIT="${LIMIT:-1}"
WORKERS="${WORKERS:-1}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-}"
EVALUATE="${EVALUATE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/vllm_qwen35/${MODEL_NAME}_${TASK_LEVEL}_$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="${BASE_CONFIG:-best_default_judge_json_agnostic.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-model_vllm_9b_base.yaml}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/home/luyadong/cred.sh}"
BROWSER_BACKEND="${BROWSER_BACKEND:-browserbase}"
START_VLLM="${START_VLLM:-1}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-12000}"
CONTEXT_RESERVE_TOKENS="${CONTEXT_RESERVE_TOKENS:-1024}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-$((MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - CONTEXT_RESERVE_TOKENS))}"
SLIDING_WINDOW_KEEP_TURNS="${SLIDING_WINDOW_KEEP_TURNS:-10}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$OUTPUT_DIR/vllm.log}"

if (( MAX_CONTEXT_TOKENS <= 0 )); then
  echo "[error] MAX_MODEL_LEN must be greater than MAX_OUTPUT_TOKENS" >&2
  exit 2
fi

if [[ -f "$CREDENTIALS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CREDENTIALS_FILE"
fi
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-${OPENAI_GATEWAY_ENDPOINT:-http://gateway.phyagi.net/api/responses}}"

mkdir -p "$OUTPUT_DIR"
VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
  command -v "$VLLM_BIN" >/dev/null 2>&1 || {
    echo "[error] vLLM executable not found: $VLLM_BIN" >&2
    exit 1
  }
  read -r -a EXTRA_VLLM_ARGS <<< "${VLLM_ARGS:-}"
  echo "[vllm] serving $MODEL_ID as $MODEL_NAME on $HOST:$PORT"
  "$VLLM_BIN" serve "$MODEL_ID" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    "${EXTRA_VLLM_ARGS[@]}" >"$VLLM_LOG_FILE" 2>&1 &
  VLLM_PID=$!

  "$PY" - "$VLLM_PID" <<PY
import sys
import time
import urllib.request
from pathlib import Path

pid = int(sys.argv[1])
url = "http://${HOST}:${PORT}/v1/models"
deadline = time.time() + int("${VLLM_WAIT_SECONDS:-900}")
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print("[vllm] ready")
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists() or stat_path.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            print("[error] vLLM exited before becoming ready", file=sys.stderr)
            raise SystemExit(1)
        time.sleep(5)
print(f"[error] vLLM did not become ready before deadline: {url}", file=sys.stderr)
raise SystemExit(1)
PY
fi

CONFIG_ARGS=(-c "$BASE_CONFIG" -c "$MODEL_CONFIG")
if [[ -n "${EXTRA_CONFIGS:-}" ]]; then
  IFS=',' read -r -a EXTRA_CONFIG_LIST <<< "$EXTRA_CONFIGS"
  for config in "${EXTRA_CONFIG_LIST[@]}"; do
    config="${config#"${config%%[![:space:]]*}"}"
    config="${config%"${config##*[![:space:]]}"}"
    [[ -n "$config" ]] && CONFIG_ARGS+=(-c "$config")
  done
fi

EVALUATE_ARGS=(--evaluate)
[[ "$EVALUATE" == "1" ]] || EVALUATE_ARGS=(--no-evaluate)
JUDGE_ENDPOINT_ARGS=()
[[ -n "$JUDGE_ENDPOINT" ]] && JUDGE_ENDPOINT_ARGS=(--judge-endpoint "$JUDGE_ENDPOINT")

echo "[eval] endpoint=$ENDPOINT model=$MODEL_NAME output=$OUTPUT_DIR"
cd "$REPO"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m miniswewebagent.run.benchmarks.om2w \
  "${CONFIG_ARGS[@]}" \
  -c "model.endpoint=$ENDPOINT" \
  -c "model.model_name=$MODEL_NAME" \
  -c "model.max_output_tokens=$MAX_OUTPUT_TOKENS" \
  -c "model.max_context_tokens=$MAX_CONTEXT_TOKENS" \
  -c "model.sliding_window_keep_turns=$SLIDING_WINDOW_KEEP_TURNS" \
  -c "environment.credentials_file=$CREDENTIALS_FILE" \
  -c "environment.env.PYTHONPATH=$REPO/agent_runtime" \
  -c "environment.env.MWA_BROWSER_BACKEND=$BROWSER_BACKEND" \
  -c "environment.env.OPENAI_COMPATIBLE_ENDPOINT=$ENDPOINT" \
  -c "environment.env.OPENAI_COMPATIBLE_MODEL=$MODEL_NAME" \
  -c "environment.env.OPENAI_COMPATIBLE_API_KEY=dummy" \
  --tasks-file "$TASKS_FILE" \
  --task-level "$TASK_LEVEL" \
  --limit "$LIMIT" \
  --workers "$WORKERS" \
  --judge-runs "$JUDGE_RUNS" \
  --judge-model "$JUDGE_MODEL" \
  --judge-python "$PY" \
  --judge-script "$REPO/om2w_judge/run.py" \
  "${JUDGE_ENDPOINT_ARGS[@]}" \
  "${EVALUATE_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
