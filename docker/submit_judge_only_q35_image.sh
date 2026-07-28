#!/usr/bin/env bash
# 提交单节点 0 GPU 的 CPU job,对 PVC 上已存在的轨迹目录单跑 WebJudge
# (模式 C,见 .claude/skills/web-agent-dist-train-eval;in-pod 逻辑在
# docker/run_judge_only_q35_image.sh)。
#
# 用法(路径按 pod 内视角,PVC 根挂在 /mnt/pvc):
#   EVAL_RUN_ID=base4b_all_run1 bash docker/submit_judge_only_q35_image.sh
#   # 等价于 TRAJECTORIES_DIR=$DATA_ROOT/evals/base4b_all_run1/outputs
#   #        JUDGE_OUTPUT_DIR=$DATA_ROOT/evals/base4b_all_run1/webjudge_partial
#   # 也可显式给 TRAJECTORIES_DIR / JUDGE_OUTPUT_DIR 覆盖
#   # 断点续判:同一 JUDGE_OUTPUT_DIR 重提即可(跳过已判 task_id)

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"

EVAL_RUN_ID="${EVAL_RUN_ID:-}"
TRAJECTORIES_DIR="${TRAJECTORIES_DIR:-}"
JUDGE_OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-}"
[[ -n "$EVAL_RUN_ID" || ( -n "$TRAJECTORIES_DIR" && -n "$JUDGE_OUTPUT_DIR" ) ]] || {
  echo "[error] set EVAL_RUN_ID, or both TRAJECTORIES_DIR and JUDGE_OUTPUT_DIR"; exit 1; }

JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-3}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-32}"
TASK_LEVEL="${TASK_LEVEL:-all}"
LIMIT="${LIMIT:-0}"
JUDGE_CPU="${JUDGE_CPU:-32}"
JUDGE_MEMORY="${JUDGE_MEMORY:-64Gi}"
JUDGE_SHM="${JUDGE_SHM:-8Gi}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/data/t-yifeili/webchain_sampling/cred.sh}"
CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-t-yifeili-webchain-sampling-creds}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

[[ -d "$MINI_WEB_AGENT_DIR" ]] || { echo "[error] missing $MINI_WEB_AGENT_DIR"; exit 1; }
[[ -f "$CREDENTIALS_FILE" ]] || { echo "[error] credentials file not found: $CREDENTIALS_FILE"; exit 1; }
[[ -d "$MINI_WEB_AGENT_DIR/om2w_judge/methods" ]] || {
  echo "[error] vendored judge missing: om2w_judge/methods"; exit 1; }

export PATH="$HOME/.krew/bin:$PATH"
export NAMESPACE="${NAMESPACE:-bonete61}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"
USER_ALIAS="${USER_ALIAS:-${USER%@*}}"
export JOB_NAME="${JOB_NAME:-${USER_ALIAS}-${PRIORITY}-${PROJECT_NAME}-webjudge}"

echo "[submit_judge_only] run_id=${EVAL_RUN_ID:-<explicit dirs>} trajectories=${TRAJECTORIES_DIR:-<from run_id>}"
echo "[submit_judge_only] judge=$JUDGE_MODEL threshold=$SCORE_THRESHOLD workers=$JUDGE_NUM_WORKERS level=$TASK_LEVEL"
echo "[submit_judge_only] cpu=$JUDGE_CPU mem=$JUDGE_MEMORY (0 GPU) PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME"

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/judge-only-upload.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() { rm -rf "$STAGING_PARENT"; }
trap cleanup_staging EXIT

mkdir -p "$UPLOAD_DIR"
for relpath in src scripts docker om2w_judge; do
  cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
for relpath in pyproject.toml README.md LICENSE; do
  [[ -f "$MINI_WEB_AGENT_DIR/$relpath" ]] && cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
echo "[submit_judge_only] staged slim upload at $UPLOAD_DIR ($(du -sh "$UPLOAD_DIR" | cut -f1))"

kubectl -n "$NAMESPACE" create secret generic "$CREDENTIALS_SECRET" \
  --from-file=cred.sh="$CREDENTIALS_FILE" \
  --dry-run=client -o yaml | kubectl -n "$NAMESPACE" apply -f -

EXTRA_ENV="JUDGE_MODEL=${JUDGE_MODEL},SCORE_THRESHOLD=${SCORE_THRESHOLD},JUDGE_NUM_WORKERS=${JUDGE_NUM_WORKERS}"
EXTRA_ENV="${EXTRA_ENV},TASK_LEVEL=${TASK_LEVEL},LIMIT=${LIMIT}"
[[ -n "$EVAL_RUN_ID" ]]       && EXTRA_ENV="${EXTRA_ENV},EVAL_RUN_ID=${EVAL_RUN_ID}"
[[ -n "$TRAJECTORIES_DIR" ]]  && EXTRA_ENV="${EXTRA_ENV},TRAJECTORIES_DIR=${TRAJECTORIES_DIR}"
[[ -n "$JUDGE_OUTPUT_DIR" ]]  && EXTRA_ENV="${EXTRA_ENV},JUDGE_OUTPUT_DIR=${JUDGE_OUTPUT_DIR}"

FOLLOW_ARGS=()
[[ "$FOLLOW_LOGS" == "1" ]] && FOLLOW_ARGS=(--follow-logs)

bash "$SUBMIT" \
  --upload "$UPLOAD_DIR" \
  --image "$IMAGE" \
  --node 1 --gpu-per-node 0 \
  --cpu "$JUDGE_CPU" --memory "$JUDGE_MEMORY" --shm "$JUDGE_SHM" \
  --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" \
  --extra-env-vars "$EXTRA_ENV" \
  "${FOLLOW_ARGS[@]}" \
  --cmd 'exec bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/docker/run_judge_only_q35_image.sh'
