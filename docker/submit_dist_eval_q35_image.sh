#!/usr/bin/env bash
# 提交一个多节点数据并行的 OM2W harness eval job(mini-web-agent 自带 harness,
# vllm serve 本地推理,非 SkyRL)。每个节点各起一个 tp=8 的 vLLM + 跑 1/NODES 的
# 任务分片;master 最后统一 judge + 汇总。支持断点续评:
#   同一 EVAL_RUN_ID 重新提交 = 跳过已完成任务接着跑(kill/断掉都不怕)。
#
# 用法:
#   EVAL_CKPT=/mnt/pvc/<alias>/models/... NODES=4 bash docker/submit_dist_eval_q35_image.sh
#   # PhiTrain WebWright checkpoint: add REQUIRE_RUNTIME_MANIFEST=1
#   # 断点续评(job 被杀后):原样重跑同一条命令(EVAL_RUN_ID 不变即可)
#   # 重跑失败任务:RETRY_FAILED=1 加在前面
#
# 并发提示:总 browserbase 并发 = NODES * WORKERS。TOTAL_WORKERS(默认 80,已验证
# 的安全水位)会自动均分到各节点;确认配额够以前不要随意加大。

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia-26.06-pytorch-2.12.1-torchao-0.17.0-te-2.16.1-deepspeed-0.19.2-fa2-1f7ce2f-fa4-4.0.0b19-vllm-0.24.0:20260707}"
NODES="${NODES:-1}"
GPUS="${GPUS:-8}"

EVAL_CKPT="${EVAL_CKPT:?set EVAL_CKPT to an HF-format ckpt dir on the PVC}"
MODEL_NAME="${MODEL_NAME:-$(basename "$EVAL_CKPT")}"
TASK_LEVEL="${TASK_LEVEL:-all}"
LIMIT="${LIMIT:-0}"
# 全局 agent 并发(浏览器会话数),均分到各节点
TOTAL_WORKERS="${TOTAL_WORKERS:-80}"
WORKERS="${WORKERS:-$(( (TOTAL_WORKERS + NODES - 1) / NODES ))}"
# 稳定 run id:同 id 重提 = 断点续评。默认 <model>_<level> 加当天日期。
EVAL_RUN_ID="${EVAL_RUN_ID:-${MODEL_NAME}_${TASK_LEVEL}_$(date +%Y%m%d)}"
RETRY_FAILED="${RETRY_FAILED:-0}"

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/om2w_sft_state_debug_vllm_sft_ckpt.yaml}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
JUDGE_SCORE_THRESHOLD="${JUDGE_SCORE_THRESHOLD:-3}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://gateway.phyagi.net/api/responses}"
TP="${TP:-8}"
# Leave semantic inference settings unset by default. Manifest-bearing
# checkpoints resolve them from web_agent_runtime.json in the in-pod preflight.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-}"
SLIDING_WINDOW_KEEP_TURNS="${SLIDING_WINDOW_KEEP_TURNS:-}"
ALLOW_TRAINING_CONTRACT_OVERRIDE="${ALLOW_TRAINING_CONTRACT_OVERRIDE:-0}"
REQUIRE_RUNTIME_MANIFEST="${REQUIRE_RUNTIME_MANIFEST:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/data/t-yifeili/webchain_sampling/cred.sh}"
CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-t-yifeili-webchain-sampling-creds}"
EVAL_CPU="${EVAL_CPU:-64}"
EVAL_MEMORY="${EVAL_MEMORY:-256Gi}"
EVAL_SHM="${EVAL_SHM:-64Gi}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

[[ -d "$MINI_WEB_AGENT_DIR" ]] || { echo "[error] missing $MINI_WEB_AGENT_DIR"; exit 1; }
[[ -f "$MINI_WEB_AGENT_DIR/src/miniswewebagent/config/$BENCHMARK_CONFIG" ]] || {
  echo "[error] benchmark config not found: src/miniswewebagent/config/$BENCHMARK_CONFIG"; exit 1; }
[[ -f "$CREDENTIALS_FILE" ]] || { echo "[error] credentials file not found: $CREDENTIALS_FILE"; exit 1; }
[[ "$REQUIRE_RUNTIME_MANIFEST" == "0" ||
   "$REQUIRE_RUNTIME_MANIFEST" == "1" ]] || {
  echo "[error] REQUIRE_RUNTIME_MANIFEST must be 0 or 1"; exit 1; }
[[ "$ALLOW_TRAINING_CONTRACT_OVERRIDE" == "0" ||
   "$ALLOW_TRAINING_CONTRACT_OVERRIDE" == "1" ]] || {
  echo "[error] ALLOW_TRAINING_CONTRACT_OVERRIDE must be 0 or 1"; exit 1; }

export PATH="$HOME/.krew/bin:$PATH"
export NAMESPACE="${NAMESPACE:-bonete61}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_dist_eval] NODES=$NODES GPUS=$GPUS IMAGE=$IMAGE"
echo "[submit_dist_eval] CKPT=$EVAL_CKPT model=$MODEL_NAME"
echo "[submit_dist_eval] run_id=$EVAL_RUN_ID (resubmit the SAME id to resume)"
echo "[submit_dist_eval] level=$TASK_LEVEL limit=$LIMIT workers=${WORKERS}/node (total ~$((WORKERS*NODES)) browserbase sessions)"
echo "[submit_dist_eval] judge=$JUDGE_MODEL threshold=$JUDGE_SCORE_THRESHOLD workers=$JUDGE_NUM_PROC endpoint=$JUDGE_ENDPOINT"
echo "[submit_dist_eval] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME"

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/dist-eval-upload.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() { rm -rf "$STAGING_PARENT"; }
trap cleanup_staging EXIT

mkdir -p "$UPLOAD_DIR"
# configs/ 里有训练对齐 chat template,必须带上
for relpath in src scripts docker om2w_judge om2w_judge_sandbox echo_rl configs; do
  cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
for relpath in pyproject.toml README.md LICENSE; do
  [[ -f "$MINI_WEB_AGENT_DIR/$relpath" ]] && cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
echo "[submit_dist_eval] staged slim upload at $UPLOAD_DIR ($(du -sh "$UPLOAD_DIR" | cut -f1))"

kubectl -n "$NAMESPACE" create secret generic "$CREDENTIALS_SECRET" \
  --from-file=cred.sh="$CREDENTIALS_FILE" \
  --dry-run=client -o yaml | kubectl -n "$NAMESPACE" apply -f -

EXTRA_ENV="EVAL_CKPT=${EVAL_CKPT},EVAL_RUN_ID=${EVAL_RUN_ID},MODEL_NAME=${MODEL_NAME}"
EXTRA_ENV="${EXTRA_ENV},BENCHMARK_CONFIG=${BENCHMARK_CONFIG},TASK_LEVEL=${TASK_LEVEL},LIMIT=${LIMIT},WORKERS=${WORKERS}"
EXTRA_ENV="${EXTRA_ENV},JUDGE_RUNS=${JUDGE_RUNS},JUDGE_NUM_PROC=${JUDGE_NUM_PROC},JUDGE_MODEL=${JUDGE_MODEL}"
EXTRA_ENV="${EXTRA_ENV},JUDGE_SCORE_THRESHOLD=${JUDGE_SCORE_THRESHOLD},RETRY_FAILED=${RETRY_FAILED}"
EXTRA_ENV="${EXTRA_ENV},TP=${TP}"
EXTRA_ENV="${EXTRA_ENV},REQUIRE_RUNTIME_MANIFEST=${REQUIRE_RUNTIME_MANIFEST}"
EXTRA_ENV="${EXTRA_ENV},ALLOW_TRAINING_CONTRACT_OVERRIDE=${ALLOW_TRAINING_CONTRACT_OVERRIDE}"
EXTRA_ENV="${EXTRA_ENV},GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
[[ -n "$MAX_MODEL_LEN" ]] && EXTRA_ENV="${EXTRA_ENV},MAX_MODEL_LEN=${MAX_MODEL_LEN}"
[[ -n "$MAX_OUTPUT_TOKENS" ]] && EXTRA_ENV="${EXTRA_ENV},MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}"
[[ -n "$MAX_CONTEXT_TOKENS" ]] && EXTRA_ENV="${EXTRA_ENV},MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS}"
[[ -n "$SLIDING_WINDOW_KEEP_TURNS" ]] &&
  EXTRA_ENV="${EXTRA_ENV},SLIDING_WINDOW_KEEP_TURNS=${SLIDING_WINDOW_KEEP_TURNS}"
[[ -n "${CHAT_TEMPLATE:-}" ]]  && EXTRA_ENV="${EXTRA_ENV},CHAT_TEMPLATE=${CHAT_TEMPLATE}"
[[ -n "${CHAT_TEMPLATE_NAME:-}" ]] && EXTRA_ENV="${EXTRA_ENV},CHAT_TEMPLATE_NAME=${CHAT_TEMPLATE_NAME}"
[[ -n "${MERGE_VISION:-}" ]]   && EXTRA_ENV="${EXTRA_ENV},MERGE_VISION=${MERGE_VISION}"
[[ -n "${ORIGINAL_JUDGE:-}" ]] && EXTRA_ENV="${EXTRA_ENV},ORIGINAL_JUDGE=${ORIGINAL_JUDGE}"
[[ -n "${PERSISTENT_JUDGE:-}" ]] && EXTRA_ENV="${EXTRA_ENV},PERSISTENT_JUDGE=${PERSISTENT_JUDGE}"
[[ -n "$JUDGE_ENDPOINT" ]] && EXTRA_ENV="${EXTRA_ENV},JUDGE_ENDPOINT=${JUDGE_ENDPOINT}"
[[ -n "${EXTRA_CONFIGS:-}" ]]  && EXTRA_ENV="${EXTRA_ENV},EXTRA_CONFIGS=${EXTRA_CONFIGS}"

FOLLOW_ARGS=()
[[ "$FOLLOW_LOGS" == "1" ]] && FOLLOW_ARGS=(--follow-logs)

bash "$SUBMIT" \
  --upload "$UPLOAD_DIR" \
  --image "$IMAGE" \
  --node "$NODES" --gpu-per-node "$GPUS" \
  --cpu "$EVAL_CPU" --memory "$EVAL_MEMORY" --shm "$EVAL_SHM" \
  --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" \
  --extra-env-vars "$EXTRA_ENV" \
  "${FOLLOW_ARGS[@]}" \
  --cmd 'exec bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/docker/run_dist_eval_q35_image.sh'
