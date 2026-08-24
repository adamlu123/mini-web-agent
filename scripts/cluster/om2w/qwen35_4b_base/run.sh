#!/usr/bin/env bash
# Serve the Qwen3.5-4B webwright-student *base* model with vLLM and run the SPB
# persistent-browser OM2W benchmark inside one Bonete GPU pod.
#
# This is the untrained baseline for scripts/cluster/om2w/qwen35_4b/run.sh: same
# harness, prompts, judge and eval config, and the same vLLM serving mode, but
# the policy is the stock student checkpoint rather than a converted PhiTrain RL
# Ray-actor checkpoint. There is nothing to convert, so STUDENT_MODEL_PATH is
# served directly.
#
# --language-model-only and the checkpoint's own chat_template.jinja are kept
# from the RL launcher on purpose: the RL actor was initialised from, and served
# under, exactly this config, so the baseline only differs in the weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../lib/cluster_runtime.sh
source "$SCRIPT_DIR/../../../lib/cluster_runtime.sh"

: "${DATA_ROOT:?DATA_ROOT is not set}"
: "${JOB_NAME:?JOB_NAME is not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID is not set}"

UPLOAD_REPO="${UPLOAD_REPO:-$DATA_ROOT/runs/$JOB_NAME/mini-web-agent}"
LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-qwen35-4b-base-eval}"
ASSET_SUBDIR="${ASSET_SUBDIR:-cluster_eval_assets}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/mnt/pvc/experiments/luyadong/models/webwright-student}"
MODEL_NAME="${MODEL_NAME:-student_base}"
TP="${TP:-8}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PORT="${PORT:-8000}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-3600}"
# Space-separated dotted `-c` overrides, appended last so they win. e.g.
#   EXTRA_CFG="agent.step_limit=50 model.max_output_tokens=32768"
EXTRA_CFG="${EXTRA_CFG:-}"

for numeric_name in TP WORKERS JUDGE_NUM_PROC MAX_MODEL_LEN PORT VLLM_WAIT_SECONDS; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen35-4b-base-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || {
    echo "[qwen35-4b-base-eval][error] invalid GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2
    exit 2
}
for required_dir in "$UPLOAD_REPO" "$STUDENT_MODEL_PATH"; do
    [[ -d "$required_dir" ]] || {
        echo "[qwen35-4b-base-eval][error] required directory is missing: $required_dir" >&2
        exit 1
    }
done
for required_name in config.json tokenizer_config.json chat_template.jinja; do
    [[ -f "$STUDENT_MODEL_PATH/$required_name" ]] || {
        echo "[qwen35-4b-base-eval][error] student model has no $required_name: $STUDENT_MODEL_PATH" >&2
        exit 1
    }
done
if [[ ! -f "$STUDENT_MODEL_PATH/model.safetensors" ]] &&
    [[ ! -f "$STUDENT_MODEL_PATH/model.safetensors.index.json" ]]; then
    echo "[qwen35-4b-base-eval][error] student model has no safetensors weights: $STUDENT_MODEL_PATH" >&2
    exit 1
fi
[[ -f "$CREDS_FILE" ]] || {
    echo "[qwen35-4b-base-eval][error] credentials file is missing: $CREDS_FILE" >&2
    exit 1
}
case "$LOCAL_REPO" in
    /tmp/mini-web-agent-qwen35-4b-base-eval*) ;;
    *)
        echo "[qwen35-4b-base-eval][error] refusing unsafe local staging path: $LOCAL_REPO" >&2
        exit 1
        ;;
esac

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
        echo "[qwen35-4b-base-eval][error] staged asset is missing: $required_file" >&2
        exit 1
    }
done

mwa_verify_sha256 qwen35-4b-base-eval "$CONFIG_FILE" "${CONFIG_SHA256:-}"
# The overlay is optional: the wrapper only stages one when OVERLAY_SOURCE is set.
OVERLAY_SPECS=()
OVERLAY_ARGS=()
if [[ -f "$OVERLAY_FILE" ]]; then
    mwa_verify_sha256 qwen35-4b-base-eval "$OVERLAY_FILE" "${OVERLAY_SHA256:-}"
    OVERLAY_SPECS=("$OVERLAY_FILE")
    OVERLAY_ARGS=(-c "$OVERLAY_FILE")
fi
GOLD_SPECS=()
GOLD_ARGS=()
if [[ -d "$GOLD_DIR" ]]; then
    GOLD_SPECS=("run.gold_trajectory_dir=$GOLD_DIR")
    GOLD_ARGS=(-c "run.gold_trajectory_dir=$GOLD_DIR")
fi
mwa_verify_sha256 qwen35-4b-base-eval "$TASKS_FILE" "${TASKS_SHA256:-}"

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

echo "[qwen35-4b-base-eval] installing harness dependencies"
# The cluster image provides most dependencies through Debian or the base Python
# environment. Installing the full dependency graph makes pip try to uninstall
# those record-less packages (notably Debian's Pygments), so install only what
# this non-interactive benchmark imports. No phitrain: nothing is converted here.
python -m pip install --no-deps -e "$REPO"
python -m pip install --no-deps browserbase playwright pyee openai pillow backoff
python -c 'import backoff, browserbase, openai, PIL, playwright, miniswewebagent'

EXTRA_CFG_SPECS=()
if [[ -n "$EXTRA_CFG" ]]; then
    read -r -a EXTRA_CFG_SPECS <<<"$EXTRA_CFG"
fi
EXTRA_CFG_ARGS=()
for spec in ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}; do
    EXTRA_CFG_ARGS+=(-c "$spec")
done

mwa_print_effective_config \
    qwen35-4b-base-eval "$CONFIG_FILE" \
    ${OVERLAY_SPECS+"${OVERLAY_SPECS[@]}"} \
    ${GOLD_SPECS+"${GOLD_SPECS[@]}"} \
    ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}

echo "[qwen35-4b-base-eval] model=$STUDENT_MODEL_PATH tp=$TP max_model_len=$MAX_MODEL_LEN"
echo "[qwen35-4b-base-eval] run_id=$EVAL_RUN_ID tasks=$TASKS_FILE workers=$WORKERS"
echo "[qwen35-4b-base-eval] overlay=${OVERLAY_SPECS[0]:-<none>}"
echo "[qwen35-4b-base-eval] gold_trajectories=$([[ -d "$GOLD_DIR" ]] && find "$GOLD_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l || echo 0)"
nvidia-smi -L

VLLM_LOG="$LOGS_DIR/vllm.log"
# Flags mirror the RL launcher so the baseline differs only in weights:
# --language-model-only resolves the multimodal config down to its text config,
# which is what the RL actor's weights describe and how PhiTrain's rollout
# engine served them. No --reasoning-parser on purpose: it moves the think block
# into `reasoning_content` and strips it from `content`, and the sft_state
# history replay stores `content` as extra.raw_response -- every past thought
# would be lost from the replayed context.
vllm serve "$STUDENT_MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --language-model-only \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --chat-template "$STUDENT_MODEL_PATH/chat_template.jinja" \
    --enable-prefix-caching \
    --trust-remote-code >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    mwa_stop_process "${VLLM_PID:-}"
}
trap cleanup EXIT

if ! mwa_wait_for_vllm \
    qwen35-4b-base-eval "$VLLM_PID" "$PORT" "$VLLM_WAIT_SECONDS"; then
    tail -200 "$VLLM_LOG" >&2 || true
    exit 1
fi

ENDPOINT="http://127.0.0.1:$PORT/v1"
echo "[qwen35-4b-base-eval] starting OM2W generation and judge"
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

echo "[qwen35-4b-base-eval] finished rc=$EVAL_RC run_root=$RUN_ROOT"
exit "$EVAL_RC"
