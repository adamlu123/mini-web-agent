#!/usr/bin/env bash
# Run the frozen Qwen3.6-27B last-observation OM2W configuration inside one
# Bonete GPU pod. The submission wrapper stages this repository and the frozen
# config/task assets under DATA_ROOT/runs/JOB_NAME/mini-web-agent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../lib/cluster_runtime.sh
source "$SCRIPT_DIR/../../../lib/cluster_runtime.sh"

: "${DATA_ROOT:?DATA_ROOT is not set}"
: "${JOB_NAME:?JOB_NAME is not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID is not set}"

UPLOAD_REPO="${UPLOAD_REPO:-$DATA_ROOT/runs/$JOB_NAME/mini-web-agent}"
LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-qwen36-eval}"
ASSET_SUBDIR="${ASSET_SUBDIR:-cluster_eval_assets}"
MODEL_ID="${MODEL_ID:-/mnt/pvc/experiments/luyadong/models/webwright-teacher}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
TP="${TP:-8}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PORT="${PORT:-8000}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-3600}"
# Empty keeps the frozen config's agent.step_limit; set to override it.
STEP_LIMIT="${STEP_LIMIT:-}"
STEP_LIMIT_ARGS=()
if [[ -n "$STEP_LIMIT" ]]; then
    [[ "$STEP_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen36-eval][error] STEP_LIMIT must be a positive integer: $STEP_LIMIT" >&2
        exit 2
    }
    STEP_LIMIT_ARGS=(-c "agent.step_limit=$STEP_LIMIT")
fi
# Empty keeps the frozen config's agent.max_context_tokens (0 = eviction off).
# Set to enable token eviction; budget should leave room for max_output_tokens
# inside the served max_model_len.
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-}"
if [[ -n "$MAX_CONTEXT_TOKENS" ]]; then
    [[ "$MAX_CONTEXT_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen36-eval][error] MAX_CONTEXT_TOKENS must be a positive integer: $MAX_CONTEXT_TOKENS" >&2
        exit 2
    }
    STEP_LIMIT_ARGS+=(-c "agent.max_context_tokens=$MAX_CONTEXT_TOKENS")
fi

for numeric_name in TP WORKERS JUDGE_NUM_PROC MAX_MODEL_LEN PORT VLLM_WAIT_SECONDS; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen36-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || {
    echo "[qwen36-eval][error] invalid GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2
    exit 2
}
[[ -d "$UPLOAD_REPO" ]] || {
    echo "[qwen36-eval][error] staged repository is missing: $UPLOAD_REPO" >&2
    exit 1
}
if [[ "$MODEL_ID" == /* && ! -d "$MODEL_ID" ]]; then
    echo "[qwen36-eval][error] PVC model directory is missing: $MODEL_ID" >&2
    exit 1
fi
[[ -f "$CREDS_FILE" ]] || {
    echo "[qwen36-eval][error] credentials file is missing: $CREDS_FILE" >&2
    exit 1
}

mwa_copy_staged_repo "$UPLOAD_REPO" "$LOCAL_REPO"
REPO="$LOCAL_REPO"
ASSET_DIR="$REPO/$ASSET_SUBDIR"
CONFIG_FILE="$ASSET_DIR/merged_config.yaml"
TASKS_FILE="$ASSET_DIR/tasks.json"
CHAT_TEMPLATE="$ASSET_DIR/qwen3_5_train_aligned.jinja"

for required_file in "$CONFIG_FILE" "$TASKS_FILE" "$CHAT_TEMPLATE"; do
    [[ -f "$required_file" ]] || {
        echo "[qwen36-eval][error] staged asset is missing: $required_file" >&2
        exit 1
    }
done

mwa_verify_sha256 qwen36-eval "$CONFIG_FILE" "${CONFIG_SHA256:-}"
mwa_verify_sha256 qwen36-eval "$TASKS_FILE" "${TASKS_SHA256:-}"
mwa_verify_sha256 qwen36-eval "$CHAT_TEMPLATE" "${CHAT_TEMPLATE_SHA256:-}"

RUN_ROOT="$DATA_ROOT/evals/$EVAL_RUN_ID"
OUTPUTS_DIR="$RUN_ROOT/outputs"
LOGS_DIR="$RUN_ROOT/logs"
mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR"
cp "$ASSET_DIR/provenance.json" "$RUN_ROOT/provenance.json"

source "$CREDS_FILE"
if [[ -n "${PHYAGI_API_KEY:-}" ]]; then
    export OPENAI_GATEWAY_API_KEY="${OPENAI_GATEWAY_API_KEY:-$PHYAGI_API_KEY}"
fi

export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/agent_runtime${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME"

echo "[qwen36-eval] installing harness dependencies"
# The cluster image provides most dependencies through Debian or the base
# Python environment. Installing the full dependency graph can make pip try to
# uninstall those record-less packages (notably Debian's Pygments), so install
# only the packages this non-interactive benchmark imports.
python -m pip install --no-deps -e "$REPO"
python -m pip install --no-deps browserbase playwright pyee openai pillow backoff
python -c 'import backoff, browserbase, openai, PIL, playwright, miniswewebagent'

EFFECTIVE_CONFIG_OVERRIDES=()
for ((index = 1; index < ${#STEP_LIMIT_ARGS[@]}; index += 2)); do
    EFFECTIVE_CONFIG_OVERRIDES+=("${STEP_LIMIT_ARGS[index]}")
done
mwa_print_effective_config \
    qwen36-eval "$CONFIG_FILE" "${EFFECTIVE_CONFIG_OVERRIDES[@]}"

echo "[qwen36-eval] model=$MODEL_ID tp=$TP max_model_len=$MAX_MODEL_LEN"
echo "[qwen36-eval] run_id=$EVAL_RUN_ID tasks=$TASKS_FILE workers=$WORKERS"
nvidia-smi -L

VLLM_LOG="$LOGS_DIR/vllm.log"
vllm serve "$MODEL_ID" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --chat-template "$CHAT_TEMPLATE" \
    --enable-prefix-caching \
    --trust-remote-code >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    mwa_stop_process "${VLLM_PID:-}"
}
trap cleanup EXIT

if ! mwa_wait_for_vllm \
    qwen36-eval "$VLLM_PID" "$PORT" "$VLLM_WAIT_SECONDS"; then
    tail -200 "$VLLM_LOG" >&2 || true
    exit 1
fi

ENDPOINT="http://127.0.0.1:$PORT/v1"
echo "[qwen36-eval] starting OM2W generation and judge"
cd "$REPO"
set +e
python -m miniswewebagent.run.benchmarks.om2w \
    -c "$CONFIG_FILE" \
    -c "model.endpoint=$ENDPOINT" \
    -c "model.model_name=$MODEL_NAME" \
    -c "environment.credentials_file=$CREDS_FILE" \
    -c "environment.env.PYTHONPATH=$REPO/agent_runtime" \
    -c "run.logs_root=$LOGS_DIR" \
    "${STEP_LIMIT_ARGS[@]}" \
    --tasks-file "$TASKS_FILE" \
    --task-level all \
    --workers "$WORKERS" \
    --judge-num-proc "$JUDGE_NUM_PROC" \
    --output-dir "$OUTPUTS_DIR" 2>&1 | tee "$LOGS_DIR/run.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

echo "[qwen36-eval] finished rc=$EVAL_RC run_root=$RUN_ROOT"
exit "$EVAL_RC"
