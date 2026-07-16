#!/usr/bin/env bash
# 提交一个多节点数据并行的 OM2W harness eval job(mini-web-agent 自带 harness,
# vllm serve 本地推理,非 SkyRL)。每个节点各起一个 tp=8 的 vLLM + 跑 1/NODES 的
# 任务分片;master 最后统一 judge + 汇总。支持断点续评:
#   同一 EVAL_RUN_ID 重新提交 = 跳过已完成任务接着跑(kill/断掉都不怕)。
#
# 用法:
#   EVAL_CKPT=/mnt/pvc/<alias>/models/... NODES=4 bash docker/submit_dist_eval_q35_image.sh
#   # 断点续评(job 被杀后):原样重跑同一条命令(EVAL_RUN_ID 不变即可)
#   # 重跑失败任务:RETRY_FAILED=1 加在前面
#
# 并发提示:总 browserbase 并发 = NODES * WORKERS。TOTAL_WORKERS(默认 80,已验证
# 的安全水位)会自动均分到各节点;确认配额够以前不要随意加大。

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
NODES="${NODES:-1}"
JUDGE_ONLY="${JUDGE_ONLY:-0}"
if [[ "$JUDGE_ONLY" == "1" ]]; then
  GPUS="${GPUS:-0}"
else
  GPUS="${GPUS:-8}"
fi

EVAL_CKPT="${EVAL_CKPT:-}"
if [[ "$JUDGE_ONLY" != "1" && -z "$EVAL_CKPT" ]]; then
  echo "[error] set EVAL_CKPT to an HF-format ckpt dir on the PVC"
  exit 1
fi
TRAJECTORIES_DIR="${TRAJECTORIES_DIR:-}"
MODEL_NAME="${MODEL_NAME:-${EVAL_CKPT:+$(basename "$EVAL_CKPT")}}"
TASK_LEVEL="${TASK_LEVEL:-all}"
LIMIT="${LIMIT:-0}"
# 全局 agent 并发(浏览器会话数),均分到各节点
TOTAL_WORKERS="${TOTAL_WORKERS:-80}"
WORKERS="${WORKERS:-$(( (TOTAL_WORKERS + NODES - 1) / NODES ))}"
# 稳定 run id:同 id 重提 = 断点续评。默认 <model>_<level> 加当天日期。
EVAL_RUN_ID="${EVAL_RUN_ID:-${MODEL_NAME:-om2w}_${TASK_LEVEL}_$(date +%Y%m%d)}"
RETRY_FAILED="${RETRY_FAILED:-0}"

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/om2w_sft_state_debug_vllm_sft_ckpt.yaml}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
JUDGE_SCORE_THRESHOLD="${JUDGE_SCORE_THRESHOLD:-3}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://gateway.phyagi.net/api/responses}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-48000}"
SLIDING_WINDOW_KEEP_TURNS="${SLIDING_WINDOW_KEEP_TURNS:-10}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/data/t-yifeili/webchain_sampling/cred.sh}"
CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-t-yifeili-webchain-sampling-creds}"
if [[ "$JUDGE_ONLY" == "1" ]]; then
  EVAL_CPU="${EVAL_CPU:-32}"
  EVAL_MEMORY="${EVAL_MEMORY:-64Gi}"
  EVAL_SHM="${EVAL_SHM:-8Gi}"
else
  EVAL_CPU="${EVAL_CPU:-64}"
  EVAL_MEMORY="${EVAL_MEMORY:-256Gi}"
  EVAL_SHM="${EVAL_SHM:-64Gi}"
fi
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

[[ -d "$MINI_WEB_AGENT_DIR" ]] || { echo "[error] missing $MINI_WEB_AGENT_DIR"; exit 1; }
[[ -f "$MINI_WEB_AGENT_DIR/src/miniswewebagent/config/$BENCHMARK_CONFIG" ]] || {
  echo "[error] benchmark config not found: src/miniswewebagent/config/$BENCHMARK_CONFIG"; exit 1; }
[[ -f "$CREDENTIALS_FILE" ]] || { echo "[error] credentials file not found: $CREDENTIALS_FILE"; exit 1; }
if [[ "$JUDGE_ONLY" == "1" ]]; then
  [[ "$NODES" == "1" ]] || { echo "[error] JUDGE_ONLY=1 requires NODES=1"; exit 1; }
  [[ -n "$TRAJECTORIES_DIR" ]] || {
    echo "[error] JUDGE_ONLY=1 requires TRAJECTORIES_DIR"; exit 1; }
fi

export PATH="$HOME/.krew/bin:$PATH"
export NAMESPACE="${NAMESPACE:-bonete61}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_dist_eval] NODES=$NODES GPUS=$GPUS IMAGE=$IMAGE"
echo "[submit_dist_eval] mode=$([[ "$JUDGE_ONLY" == "1" ]] && echo judge-only || echo generate-and-judge)"
if [[ "$JUDGE_ONLY" == "1" ]]; then
  echo "[submit_dist_eval] trajectories=$TRAJECTORIES_DIR"
else
  echo "[submit_dist_eval] CKPT=$EVAL_CKPT model=$MODEL_NAME"
fi
echo "[submit_dist_eval] run_id=$EVAL_RUN_ID (resubmit the SAME id to resume)"
if [[ "$JUDGE_ONLY" == "1" ]]; then
  echo "[submit_dist_eval] level=$TASK_LEVEL limit=$LIMIT generation=disabled"
else
  echo "[submit_dist_eval] level=$TASK_LEVEL limit=$LIMIT workers=${WORKERS}/node (total ~$((WORKERS*NODES)) browserbase sessions)"
fi
echo "[submit_dist_eval] judge=$JUDGE_MODEL threshold=$JUDGE_SCORE_THRESHOLD workers=$JUDGE_NUM_PROC endpoint=$JUDGE_ENDPOINT"
echo "[submit_dist_eval] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME"

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/dist-eval-upload.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() { rm -rf "$STAGING_PARENT"; }
trap cleanup_staging EXIT

mkdir -p "$UPLOAD_DIR"
# configs/ 里有训练对齐 chat template,必须带上
for relpath in src scripts docker om2w_judge echo_rl configs; do
  cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
for relpath in pyproject.toml README.md LICENSE; do
  [[ -f "$MINI_WEB_AGENT_DIR/$relpath" ]] && cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
echo "[submit_dist_eval] staged slim upload at $UPLOAD_DIR ($(du -sh "$UPLOAD_DIR" | cut -f1))"

kubectl -n "$NAMESPACE" create secret generic "$CREDENTIALS_SECRET" \
  --from-file=cred.sh="$CREDENTIALS_FILE" \
  --dry-run=client -o yaml | kubectl -n "$NAMESPACE" apply -f -

EXTRA_ENV="EVAL_RUN_ID=${EVAL_RUN_ID},MODEL_NAME=${MODEL_NAME},JUDGE_ONLY=${JUDGE_ONLY}"
[[ -n "$EVAL_CKPT" ]] && EXTRA_ENV="${EXTRA_ENV},EVAL_CKPT=${EVAL_CKPT}"
[[ -n "$TRAJECTORIES_DIR" ]] && EXTRA_ENV="${EXTRA_ENV},TRAJECTORIES_DIR=${TRAJECTORIES_DIR}"
EXTRA_ENV="${EXTRA_ENV},BENCHMARK_CONFIG=${BENCHMARK_CONFIG},TASK_LEVEL=${TASK_LEVEL},LIMIT=${LIMIT},WORKERS=${WORKERS}"
EXTRA_ENV="${EXTRA_ENV},JUDGE_RUNS=${JUDGE_RUNS},JUDGE_NUM_PROC=${JUDGE_NUM_PROC},JUDGE_MODEL=${JUDGE_MODEL}"
EXTRA_ENV="${EXTRA_ENV},JUDGE_SCORE_THRESHOLD=${JUDGE_SCORE_THRESHOLD},RETRY_FAILED=${RETRY_FAILED}"
EXTRA_ENV="${EXTRA_ENV},TP=${TP},MAX_MODEL_LEN=${MAX_MODEL_LEN},MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}"
EXTRA_ENV="${EXTRA_ENV},MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS},SLIDING_WINDOW_KEEP_TURNS=${SLIDING_WINDOW_KEEP_TURNS}"
EXTRA_ENV="${EXTRA_ENV},GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
[[ -n "${CHAT_TEMPLATE:-}" ]]  && EXTRA_ENV="${EXTRA_ENV},CHAT_TEMPLATE=${CHAT_TEMPLATE}"
[[ -n "${MERGE_VISION:-}" ]]  && EXTRA_ENV="${EXTRA_ENV},MERGE_VISION=${MERGE_VISION}"
[[ -n "${BASE_MODEL_ID:-}" ]]  && EXTRA_ENV="${EXTRA_ENV},BASE_MODEL_ID=${BASE_MODEL_ID}"
[[ -n "${HF_HOME:-}" ]]        && EXTRA_ENV="${EXTRA_ENV},HF_HOME=${HF_HOME}"
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
