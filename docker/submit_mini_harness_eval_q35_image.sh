#!/usr/bin/env bash
# 提交一个 eval-only pod：用当前 mini-web-agent 代码里的历史 miniswewebagent
# harness 评测 0623 state-debug SFT checkpoint。默认 p0 / cua / 8xB200。

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
GPUS="${GPUS:-8}"
EVAL_CKPT="${EVAL_CKPT:-/mnt/pvc/${USER_ALIAS:-${USER%@*}}/models/qwen35_9b/full/web_agent_state_debug_latest_0623}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-state_debug_0623_fixed_harness}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/om2w_sft_state_debug_vllm_sft_ckpt.yaml}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/data/t-yifeili/webchain_sampling/cred.sh}"
CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-t-yifeili-webchain-sampling-creds}"
TASK_LEVEL="${TASK_LEVEL:-all}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-8}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

[[ -d "$MINI_WEB_AGENT_DIR" ]] || { echo "[error] missing $MINI_WEB_AGENT_DIR"; exit 1; }
[[ -f "$MINI_WEB_AGENT_DIR/src/miniswewebagent/config/$BENCHMARK_CONFIG" ]] || {
  echo "[error] benchmark config not found: src/miniswewebagent/config/$BENCHMARK_CONFIG"
  exit 1
}
[[ -f "$CREDENTIALS_FILE" ]] || { echo "[error] credentials file not found: $CREDENTIALS_FILE"; exit 1; }

export PATH="$HOME/.krew/bin:$PATH"
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"
export NAMESPACE="${NAMESPACE:-bonete61}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_mini_harness_eval] GPUS=$GPUS IMAGE=$IMAGE"
echo "[submit_mini_harness_eval] CKPT=$EVAL_CKPT"
echo "[submit_mini_harness_eval] benchmark=$BENCHMARK_CONFIG level=$TASK_LEVEL limit=$LIMIT workers=$WORKERS tag=$EVAL_RUN_TAG"
echo "[submit_mini_harness_eval] credentials=$CREDENTIALS_FILE -> secret/$CREDENTIALS_SECRET"
echo "[submit_mini_harness_eval] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME"

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/mini-harness-upload.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() {
  rm -rf "$STAGING_PARENT"
}
trap cleanup_staging EXIT

mkdir -p "$UPLOAD_DIR"
for relpath in src scripts docker om2w_judge echo_rl; do
  cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
for relpath in pyproject.toml README.md LICENSE; do
  [[ -f "$MINI_WEB_AGENT_DIR/$relpath" ]] && cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
echo "[submit_mini_harness_eval] staged slim upload at $UPLOAD_DIR ($(du -sh "$UPLOAD_DIR" | cut -f1))"

kubectl -n "$NAMESPACE" create secret generic "$CREDENTIALS_SECRET" \
  --from-file=cred.sh="$CREDENTIALS_FILE" \
  --dry-run=client -o yaml | kubectl -n "$NAMESPACE" apply -f -

EXTRA_ENV="EVAL_CKPT=${EVAL_CKPT},EVAL_RUN_TAG=${EVAL_RUN_TAG},BENCHMARK_CONFIG=${BENCHMARK_CONFIG},TASK_LEVEL=${TASK_LEVEL},LIMIT=${LIMIT},WORKERS=${WORKERS},JUDGE_RUNS=${JUDGE_RUNS},JUDGE_NUM_PROC=${JUDGE_NUM_PROC},TP=${TP},MAX_MODEL_LEN=${MAX_MODEL_LEN},MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS},GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
[[ -n "${MODEL_NAME:-}" ]] && EXTRA_ENV="${EXTRA_ENV},MODEL_NAME=${MODEL_NAME}"

FOLLOW_ARGS=()
if [[ "$FOLLOW_LOGS" == "1" ]]; then
  FOLLOW_ARGS=(--follow-logs)
fi

bash "$SUBMIT" \
  --upload "$UPLOAD_DIR" \
  --image "$IMAGE" \
  --node 1 --gpu-per-node "$GPUS" \
  --cpu 64 --memory 512Gi --shm 64Gi \
  --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" \
  --extra-env-vars "$EXTRA_ENV" \
  "${FOLLOW_ARGS[@]}" \
  --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_mini_harness_eval_q35_image.sh'