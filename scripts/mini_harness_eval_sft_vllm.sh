#!/usr/bin/env bash
# Run the historical miniswewebagent Online-Mind2Web harness against a Qwen3.5-9B
# base model or checkpoint served through an OpenAI-compatible local vLLM endpoint.
#
# Examples:
#   SMOKE=1 bash scripts/mini_harness_eval_sft_vllm.sh
#   SMOKE=1 CKPT=/data/t-yifeili/ckpts/eval_mt_1ep MODEL_NAME=eval_mt_1ep bash scripts/mini_harness_eval_sft_vllm.sh
#   START_VLLM=0 ENDPOINT=http://127.0.0.1:8000/v1/chat/completions MODEL_NAME=eval_mt_1ep \
#     TASK_LEVEL=easy LIMIT=5 bash scripts/mini_harness_eval_sft_vllm.sh
set -euo pipefail

REPO="${REPO:-/data/t-yifeili/mini-web-agent}"
PY="${PY:-/data/t-yifeili/miniconda3/envs/echo-rl/bin/python}"
CKPT="${CKPT:-Qwen/Qwen3.5-9B}"
if [[ "$CKPT" == "Qwen/Qwen3.5-9B" ]]; then
  DEFAULT_MODEL_NAME="qwen35_9b_base"
else
  DEFAULT_MODEL_NAME="$(basename "$CKPT")"
fi
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-http://${HOST}:${PORT}/v1/chat/completions}"
TASKS_FILE="${TASKS_FILE:-src/miniswewebagent/run/benchmarks/om2w_260220.json}"
TASK_LEVEL="${TASK_LEVEL:-easy}"
LIMIT="${LIMIT:-1}"
WORKERS="${WORKERS:-1}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/om2w_sft_vllm.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/sft_vllm/${MODEL_NAME}_${TASK_LEVEL}_$(date +%Y%m%d_%H%M%S)}"
START_VLLM="${START_VLLM:-1}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-12000}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$PY")/vllm}"
[[ -x "$VLLM_BIN" ]] || VLLM_BIN="vllm"
VLLM_LOG_TO_STDOUT="${VLLM_LOG_TO_STDOUT:-0}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$OUTPUT_DIR/vllm.log}"

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

USER_OPENAI_GATEWAY_API_KEY="${OPENAI_GATEWAY_API_KEY:-}"
USER_PHYAGI_API_KEY="${PHYAGI_API_KEY:-}"
USER_OPENAI_API_KEY="${OPENAI_API_KEY:-}"

if [[ -f /home/luyadong/cred.sh ]]; then
  # Provides Browserbase creds and judge keys on the shared machine.
  # shellcheck disable=SC1091
  source /home/luyadong/cred.sh
fi

# Keep explicitly exported user keys ahead of values loaded from cred.sh.  The
# benchmark judge resolves gateway auth as OPENAI_GATEWAY_API_KEY first, then
# PHYAGI_API_KEY, so mirror a user-provided PHYAGI key into the first slot to
# avoid a stale OPENAI_GATEWAY_API_KEY winning priority.
[[ -n "$USER_PHYAGI_API_KEY" ]] && export PHYAGI_API_KEY="$USER_PHYAGI_API_KEY"
[[ -n "$USER_OPENAI_API_KEY" ]] && export OPENAI_API_KEY="$USER_OPENAI_API_KEY"
if [[ -n "$USER_PHYAGI_API_KEY" ]]; then
  export OPENAI_GATEWAY_API_KEY="$USER_PHYAGI_API_KEY"
elif [[ -n "$USER_OPENAI_GATEWAY_API_KEY" ]]; then
  export OPENAI_GATEWAY_API_KEY="$USER_OPENAI_GATEWAY_API_KEY"
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
  mkdir -p "$(dirname "$VLLM_LOG_FILE")"
  if [[ "$VLLM_LOG_TO_STDOUT" == "1" ]]; then
    "$VLLM_BIN" serve "$CKPT" \
      --served-model-name "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --trust-remote-code \
      ${VLLM_ARGS:-} &
  else
    echo "[vllm] log: $VLLM_LOG_FILE"
    "$VLLM_BIN" serve "$CKPT" \
      --served-model-name "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --trust-remote-code \
      ${VLLM_ARGS:-} >"$VLLM_LOG_FILE" 2>&1 &
  fi
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

CONFIG_ARGS=( -c mini.yaml -c "$BENCHMARK_CONFIG" )
if [[ -n "$EXTRA_CONFIGS" ]]; then
  IFS=',' read -r -a _extra_cfgs <<< "$EXTRA_CONFIGS"
  for _cfg in "${_extra_cfgs[@]}"; do
    _cfg="${_cfg#${_cfg%%[![:space:]]*}}"
    _cfg="${_cfg%${_cfg##*[![:space:]]}}"
    [[ -n "$_cfg" ]] && CONFIG_ARGS+=( -c "$_cfg" )
  done
fi

"$PY" -m miniswewebagent.run.benchmarks.om2w \
  "${CONFIG_ARGS[@]}" \
  -c "model.endpoint=$ENDPOINT" \
  -c "model.model_name=$MODEL_NAME" \
  -c "model.max_output_tokens=$MAX_OUTPUT_TOKENS" \
  -c "environment.env.WEB_AGENT_POLICY_URL=$ENDPOINT" \
  -c "environment.env.WEB_AGENT_POLICY_MODEL=$MODEL_NAME" \
  -c "environment.env.OPENAI_COMPATIBLE_ENDPOINT=$ENDPOINT" \
  -c "environment.env.OPENAI_COMPATIBLE_MODEL=$MODEL_NAME" \
  -c "environment.env.OPENAI_COMPATIBLE_API_KEY=dummy" \
  -c "environment.env.OPENAI_GATEWAY_MODEL=$MODEL_NAME" \
  --tasks-file "$TASKS_FILE" \
  --task-level "$TASK_LEVEL" \
  --limit "$LIMIT" \
  --workers "$WORKERS" \
  --judge-runs "$JUDGE_RUNS" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
rc=$?
exit "$rc"
