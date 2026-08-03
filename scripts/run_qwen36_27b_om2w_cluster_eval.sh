#!/usr/bin/env bash
# Run the frozen Qwen3.6-27B last-observation OM2W configuration inside one
# Bonete GPU pod. The submission wrapper stages this repository and the frozen
# config/task assets under DATA_ROOT/runs/JOB_NAME/mini-web-agent.

set -euo pipefail

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

rm -rf "$LOCAL_REPO"
mkdir -p "$LOCAL_REPO"
cp -R --no-preserve=mode,ownership,timestamps "$UPLOAD_REPO/." "$LOCAL_REPO/"
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

verify_sha256() {
    local file_path="$1"
    local expected="$2"
    local actual
    [[ -z "$expected" ]] && return 0
    actual="$(sha256sum "$file_path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "[qwen36-eval][error] checksum mismatch for $file_path" >&2
        echo "[qwen36-eval][error] expected=$expected actual=$actual" >&2
        exit 1
    }
}
verify_sha256 "$CONFIG_FILE" "${CONFIG_SHA256:-}"
verify_sha256 "$TASKS_FILE" "${TASKS_SHA256:-}"
verify_sha256 "$CHAT_TEMPLATE" "${CHAT_TEMPLATE_SHA256:-}"

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
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if ! python - "$VLLM_PID" "$PORT" "$VLLM_WAIT_SECONDS" <<'PY'
import sys
import time
import urllib.request
from pathlib import Path

pid, port, timeout = map(int, sys.argv[1:])
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print(f"[qwen36-eval] vLLM ready: {url}", flush=True)
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists():
            raise SystemExit("[qwen36-eval][error] vLLM exited before readiness")
        time.sleep(5)
raise SystemExit(f"[qwen36-eval][error] vLLM readiness timed out: {url}")
PY
then
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
    --tasks-file "$TASKS_FILE" \
    --task-level all \
    --workers "$WORKERS" \
    --judge-num-proc "$JUDGE_NUM_PROC" \
    --output-dir "$OUTPUTS_DIR" 2>&1 | tee "$LOGS_DIR/run.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

echo "[qwen36-eval] finished rc=$EVAL_RC run_root=$RUN_ROOT"
exit "$EVAL_RC"
