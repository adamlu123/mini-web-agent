#!/usr/bin/env bash
# Run stock Qwen3.8-27B (https://huggingface.co/Qwen/Qwen3.8-27B) on the SPB
# persistent-browser OM2W benchmark inside one Bonete GPU pod.
#
# Same harness, prompts and judge as scripts/cluster/om2w/qwen36_27b/run.sh --
# the submission wrapper stages a base config (eval/om2w_spb_vllm_lastobs.yaml by
# default, or a frozen run snapshot when CONFIG_SOURCE is overridden). An overlay
# yaml is optional and only applied when the wrapper staged one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../lib/cluster_runtime.sh
source "$SCRIPT_DIR/../../../lib/cluster_runtime.sh"

: "${DATA_ROOT:?DATA_ROOT is not set}"
: "${JOB_NAME:?JOB_NAME is not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID is not set}"

UPLOAD_REPO="${UPLOAD_REPO:-$DATA_ROOT/runs/$JOB_NAME/mini-web-agent}"
LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-qwen38-eval}"
ASSET_SUBDIR="${ASSET_SUBDIR:-cluster_eval_assets}"
# A Hugging Face repo id (downloaded into HF_HOME on the PVC) or an absolute path.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_NAME="${MODEL_NAME:-qwen38_27b}"
TP="${TP:-8}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
# Qwen3.8 is native 262144. 131072 leaves generous headroom over the 16384-token
# thinking budget without making vLLM's memory profiling pathological.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PORT="${PORT:-8000}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-3600}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
# Empty (the default) serves the checkpoint's own chat template, which is the
# correct one for a stock instruct model. Only point this at
# qwen3_5_train_aligned.jinja for a checkpoint we fine-tuned ourselves.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
# Space-separated dotted `-c` overrides, appended last so they win. e.g.
#   EXTRA_CFG="agent.step_limit=50 model.max_output_tokens=32768"
EXTRA_CFG="${EXTRA_CFG:-}"

for numeric_name in TP WORKERS JUDGE_NUM_PROC MAX_MODEL_LEN PORT VLLM_WAIT_SECONDS DOWNLOAD_RETRIES; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen38-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || {
    echo "[qwen38-eval][error] invalid GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2
    exit 2
}
[[ -d "$UPLOAD_REPO" ]] || {
    echo "[qwen38-eval][error] staged repository is missing: $UPLOAD_REPO" >&2
    exit 1
}
[[ -f "$CREDS_FILE" ]] || {
    echo "[qwen38-eval][error] credentials file is missing: $CREDS_FILE" >&2
    exit 1
}
if [[ -n "$CHAT_TEMPLATE" && ! -f "$CHAT_TEMPLATE" ]]; then
    echo "[qwen38-eval][error] chat template is missing: $CHAT_TEMPLATE" >&2
    exit 1
fi

mwa_copy_staged_repo "$UPLOAD_REPO" "$LOCAL_REPO"
REPO="$LOCAL_REPO"
ASSET_DIR="$REPO/$ASSET_SUBDIR"
CONFIG_FILE="$ASSET_DIR/merged_config.yaml"
OVERLAY_FILE="$ASSET_DIR/overlay_config.yaml"
TASKS_FILE="$ASSET_DIR/tasks.json"
# Staged only when the submission wrapper was given GOLD_SOURCE. The config's own
# run.gold_trajectory_dir points at a submitting-host path that does not exist in
# this pod, so it is repointed here whenever the assets are present.
GOLD_DIR="$ASSET_DIR/gold"

for required_file in "$CONFIG_FILE" "$TASKS_FILE"; do
    [[ -f "$required_file" ]] || {
        echo "[qwen38-eval][error] staged asset is missing: $required_file" >&2
        exit 1
    }
done

mwa_verify_sha256 qwen38-eval "$CONFIG_FILE" "${CONFIG_SHA256:-}"
# The overlay is optional: the wrapper only stages one when OVERLAY_SOURCE is set.
OVERLAY_SPECS=()
OVERLAY_ARGS=()
if [[ -f "$OVERLAY_FILE" ]]; then
    mwa_verify_sha256 qwen38-eval "$OVERLAY_FILE" "${OVERLAY_SHA256:-}"
    OVERLAY_SPECS=("$OVERLAY_FILE")
    OVERLAY_ARGS=(-c "$OVERLAY_FILE")
fi
GOLD_SPECS=()
GOLD_ARGS=()
if [[ -d "$GOLD_DIR" ]]; then
    GOLD_SPECS=("run.gold_trajectory_dir=$GOLD_DIR")
    GOLD_ARGS=(-c "run.gold_trajectory_dir=$GOLD_DIR")
fi
mwa_verify_sha256 qwen38-eval "$TASKS_FILE" "${TASKS_SHA256:-}"

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
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/agent_runtime${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME"

echo "[qwen38-eval] installing harness dependencies"
# The cluster image provides most dependencies through Debian or the base Python
# environment. Installing the full dependency graph makes pip try to uninstall
# those record-less packages (notably Debian's Pygments), so install only what
# this non-interactive benchmark imports.
python -m pip install --no-deps -e "$REPO"
python -m pip install --no-deps browserbase playwright pyee openai pillow backoff
python -c 'import backoff, browserbase, openai, PIL, playwright, miniswewebagent'

# Resolve the policy weights before serving. Letting vLLM download lazily turns
# any hub failure into an opaque readiness timeout an hour later.
if [[ "$MODEL_ID" == /* ]]; then
    [[ -d "$MODEL_ID" ]] || {
        echo "[qwen38-eval][error] model directory is missing: $MODEL_ID" >&2
        exit 1
    }
    MODEL_PATH="$MODEL_ID"
else
    echo "[qwen38-eval] downloading $MODEL_ID into $HF_HOME"
    if python -m pip install --no-deps hf_transfer >/dev/null 2>&1; then
        export HF_HUB_ENABLE_HF_TRANSFER=1
    else
        echo "[qwen38-eval] hf_transfer unavailable; using the default downloader"
    fi
    MODEL_PATH=""
    for attempt in $(seq 1 "$DOWNLOAD_RETRIES"); do
        if MODEL_PATH="$(python - "$MODEL_ID" <<'PY'
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], max_workers=8))
PY
        )"; then
            break
        fi
        echo "[qwen38-eval] download attempt $attempt/$DOWNLOAD_RETRIES failed; retrying" >&2
        MODEL_PATH=""
        sleep 30
    done
    [[ -n "$MODEL_PATH" ]] || {
        echo "[qwen38-eval][error] could not download $MODEL_ID" >&2
        exit 1
    }
fi
echo "[qwen38-eval] model_path=$MODEL_PATH"

EXTRA_CFG_SPECS=()
if [[ -n "$EXTRA_CFG" ]]; then
    read -r -a EXTRA_CFG_SPECS <<<"$EXTRA_CFG"
fi
EXTRA_CFG_ARGS=()
for spec in ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}; do
    EXTRA_CFG_ARGS+=(-c "$spec")
done

mwa_print_effective_config \
    qwen38-eval "$CONFIG_FILE" \
    ${OVERLAY_SPECS+"${OVERLAY_SPECS[@]}"} \
    ${GOLD_SPECS+"${GOLD_SPECS[@]}"} \
    ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}

echo "[qwen38-eval] model=$MODEL_ID tp=$TP max_model_len=$MAX_MODEL_LEN"
echo "[qwen38-eval] run_id=$EVAL_RUN_ID tasks=$TASKS_FILE workers=$WORKERS"
echo "[qwen38-eval] chat_template=${CHAT_TEMPLATE:-<model default>}"
echo "[qwen38-eval] overlay=${OVERLAY_SPECS[0]:-<none>}"
echo "[qwen38-eval] gold_trajectories=$([[ -d "$GOLD_DIR" ]] && find "$GOLD_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l || echo 0)"
nvidia-smi -L

template_arg=()
[[ -n "$CHAT_TEMPLATE" ]] && template_arg=(--chat-template "$CHAT_TEMPLATE")

VLLM_LOG="$LOGS_DIR/vllm.log"
# No --reasoning-parser on purpose: it moves the think block into
# `reasoning_content` and strips it from `content`, and the sft_state history
# replay stores `content` as extra.raw_response -- every past thought would be
# lost from the replayed context.
vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    "${template_arg[@]}" \
    --enable-prefix-caching \
    --trust-remote-code >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    mwa_stop_process "${VLLM_PID:-}"
}
trap cleanup EXIT

if ! mwa_wait_for_vllm \
    qwen38-eval "$VLLM_PID" "$PORT" "$VLLM_WAIT_SECONDS"; then
    tail -200 "$VLLM_LOG" >&2 || true
    exit 1
fi

ENDPOINT="http://127.0.0.1:$PORT/v1"
echo "[qwen38-eval] starting OM2W generation and judge"
cd "$REPO"
set +e
python -m miniswewebagent.run.benchmarks.om2w \
    -c "$CONFIG_FILE" \
    ${OVERLAY_ARGS+"${OVERLAY_ARGS[@]}"} \
    -c "model.endpoint=$ENDPOINT" \
    -c "model.model_name=$MODEL_NAME" \
    -c "environment.credentials_file=$CREDS_FILE" \
    -c "environment.env.PYTHONPATH=$REPO/agent_runtime" \
    -c "run.logs_root=$LOGS_DIR" \
    ${GOLD_ARGS+"${GOLD_ARGS[@]}"} \
    ${EXTRA_CFG_ARGS+"${EXTRA_CFG_ARGS[@]}"} \
    --tasks-file "$TASKS_FILE" \
    --task-level all \
    --workers "$WORKERS" \
    --judge-num-proc "$JUDGE_NUM_PROC" \
    --output-dir "$OUTPUTS_DIR" 2>&1 | tee "$LOGS_DIR/run.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

echo "[qwen38-eval] finished rc=$EVAL_RC run_root=$RUN_ROOT"
exit "$EVAL_RC"
