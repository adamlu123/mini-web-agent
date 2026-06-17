#!/usr/bin/env bash
# Run the historical miniswewebagent Online-Mind2Web harness against an SFT
# checkpoint served through an OpenAI-compatible local vLLM endpoint.
#
# Examples:
#   SMOKE=1 CKPT=/data/t-yifeili/ckpts/eval_mt_1ep bash scripts/mini_harness_eval_sft_vllm.sh
#   START_VLLM=0 ENDPOINT=http://127.0.0.1:8000/v1/chat/completions MODEL_NAME=eval_mt_1ep \
#     TASK_LEVEL=easy LIMIT=5 bash scripts/mini_harness_eval_sft_vllm.sh
set -euo pipefail

REPO="${REPO:-/data/t-yifeili/mini-web-agent}"
PY="${PY:-/data/t-yifeili/miniconda3/envs/echo-rl/bin/python}"
CKPT="${CKPT:-/data/t-yifeili/ckpts/websft_32k}"
DEFAULT_MODEL_NAME="$(basename "$CKPT")"
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-http://${HOST}:${PORT}/v1/chat/completions}"
TASKS_FILE="${TASKS_FILE:-src/miniswewebagent/run/benchmarks/om2w_260220.json}"
TASK_LEVEL="${TASK_LEVEL:-easy}"
LIMIT="${LIMIT:-1}"
WORKERS="${WORKERS:-1}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/sft_vllm/${MODEL_NAME}_${TASK_LEVEL}_$(date +%Y%m%d_%H%M%S)}"
START_VLLM="${START_VLLM:-1}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16000}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$PY")/vllm}"
[[ -x "$VLLM_BIN" ]] || VLLM_BIN="vllm"

if [[ "${SMOKE:-0}" == "1" ]]; then
  LIMIT=1
  WORKERS=1
  JUDGE_RUNS=1
fi

if [[ -d "$CKPT" ]]; then
  ls "$CKPT"/*.safetensors >/dev/null 2>&1 || ls "$CKPT"/model.safetensors.index.json >/dev/null 2>&1 \
    || { echo "[error] $CKPT is not an HF-format weights dir (*.safetensors missing)"; exit 1; }
  [[ -f "$CKPT/chat_template.jinja" ]] || echo "[warn] $CKPT/chat_template.jinja missing; vLLM will use tokenizer defaults"
else
  echo "[info] CKPT is not a local directory; treating it as an HF model id: $CKPT"
fi

cd "$REPO"
"$PY" -m pip install -e . --no-deps >/dev/null

if [[ -f /home/luyadong/cred.sh ]]; then
  # Provides Browserbase creds and judge keys on the shared machine.
  # shellcheck disable=SC1091
  source /home/luyadong/cred.sh
fi

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
  echo "[vllm] starting: $CKPT as $MODEL_NAME on $HOST:$PORT"
  "$VLLM_BIN" serve "$CKPT" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    ${VLLM_ARGS:-} &
  VLLM_PID=$!

  echo "[vllm] waiting for $ENDPOINT"
  "$PY" - <<PY
import sys, time, urllib.request
url = "http://${HOST}:${PORT}/v1/models"
deadline = time.time() + int("${VLLM_WAIT_SECONDS:-900}")
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print("[vllm] ready")
                raise SystemExit(0)
    except Exception:
        time.sleep(5)
print(f"[error] vLLM did not become ready before deadline: {url}", file=sys.stderr)
raise SystemExit(1)
PY
fi

echo "[eval] historical mini harness"
echo "[eval] endpoint=$ENDPOINT model=$MODEL_NAME"
echo "[eval] max_output_tokens=$MAX_OUTPUT_TOKENS"
echo "[eval] tasks=$TASKS_FILE level=$TASK_LEVEL limit=$LIMIT workers=$WORKERS output=$OUTPUT_DIR"

"$PY" -m miniswewebagent.run.benchmarks.om2w \
  -c benchmark/om2w_sft_vllm.yaml \
  -c "model.endpoint=$ENDPOINT" \
  -c "model.model_name=$MODEL_NAME" \
  -c "model.max_output_tokens=$MAX_OUTPUT_TOKENS" \
  --tasks-file "$TASKS_FILE" \
  --task-level "$TASK_LEVEL" \
  --limit "$LIMIT" \
  --workers "$WORKERS" \
  --judge-runs "$JUDGE_RUNS" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
rc=$?
exit "$rc"
