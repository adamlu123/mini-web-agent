#!/usr/bin/env bash
# 模式 C(见 .claude/skills/web-agent-dist-train-eval):对已存在的轨迹目录
# 单跑原版 OM2W WebJudge 的 CPU job(0 GPU,不加载 ckpt、不起 vLLM/
# Browserbase)。用于生成 job 还没跑完/挂掉时,先看已生成轨迹的 success rate。
#
# - 只判 TRAJECTORIES_DIR 里已有 final_runs 布局的 task 目录,缺的直接 Skip,
#   不稀释分数;最后 stdout 打 "Success rate: passed/total"。
# - JUDGE_OUTPUT_DIR 与轨迹目录分开、持久化;同一目录重跑会按 JSONL 里的
#   task_id 跳过已判任务(断点续判),也不会碰生成 job 未来的 outputs_eval_1/。
# - vendored OpenaiEngine 直连 OpenAI,用 secret 里的 OPENAI_API_KEY,
#   不要传 phyagi gateway token。
#
# 必需 env(submit_job.sh 注入):DATA_ROOT JOB_NAME
# 业务 env(submit_judge_only_q35_image.sh 转发):EVAL_RUN_ID(或直接给
#   TRAJECTORIES_DIR/JUDGE_OUTPUT_DIR)JUDGE_MODEL SCORE_THRESHOLD
#   JUDGE_NUM_WORKERS TASK_LEVEL LIMIT

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT not set}"
: "${JOB_NAME:?JOB_NAME not set}"

EVAL_RUN_ID="${EVAL_RUN_ID:-}"
TRAJECTORIES_DIR="${TRAJECTORIES_DIR:-${EVAL_RUN_ID:+$DATA_ROOT/evals/$EVAL_RUN_ID/outputs}}"
JUDGE_OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-${EVAL_RUN_ID:+$DATA_ROOT/evals/$EVAL_RUN_ID/webjudge_partial}}"
: "${TRAJECTORIES_DIR:?set TRAJECTORIES_DIR or EVAL_RUN_ID}"
: "${JUDGE_OUTPUT_DIR:?set JUDGE_OUTPUT_DIR or EVAL_RUN_ID}"

JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-3}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-32}"
TASK_LEVEL="${TASK_LEVEL:-all}"
LIMIT="${LIMIT:-0}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"

UPLOAD_ROOT="$DATA_ROOT/runs/$JOB_NAME"
REPO="${REPO:-$UPLOAD_ROOT/mini-web-agent}"
if [[ -d "$REPO" ]]; then
  LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-judgecopy}"
  rm -rf "$LOCAL_REPO"
  mkdir -p "$LOCAL_REPO"
  # 不能 cp -a:PVC -> 容器 /tmp(overlayfs)保留权限会 "Operation not supported"
  cp -R --no-preserve=mode,ownership,timestamps "$REPO/." "$LOCAL_REPO/"
  REPO="$LOCAL_REPO"
fi
# eval_with_original_om2w.py 从 /home/luyadong/sandbox/mini-web-agent 硬编码
# 加载 vendored judge(om2w_judge/methods)
mkdir -p /home/luyadong/sandbox
ln -sfnT "$REPO" /home/luyadong/sandbox/mini-web-agent

echo "[judge-only] job=$JOB_NAME host=$(hostname)"
echo "[judge-only] trajectories=$TRAJECTORIES_DIR"
echo "[judge-only] judge_output=$JUDGE_OUTPUT_DIR"
echo "[judge-only] model=$JUDGE_MODEL threshold=$SCORE_THRESHOLD workers=$JUDGE_NUM_WORKERS level=$TASK_LEVEL limit=$LIMIT"

[[ -f "$CREDS_FILE" ]] || { echo "[judge-only][error] credentials file not found: $CREDS_FILE"; exit 1; }
[[ -d "$TRAJECTORIES_DIR" ]] || { echo "[judge-only][error] trajectories dir not found: $TRAJECTORIES_DIR"; exit 1; }
[[ -d "$REPO/om2w_judge/methods" ]] || { echo "[judge-only][error] vendored judge missing: $REPO/om2w_judge/methods"; exit 1; }

N_TRAJ="$(find "$TRAJECTORIES_DIR" -mindepth 2 -maxdepth 2 -type d -name final_runs | wc -l)"
echo "[judge-only] $N_TRAJ task dir(s) currently have final_runs/"
(( N_TRAJ > 0 )) || { echo "[judge-only][error] no final_runs under $TRAJECTORIES_DIR yet"; exit 1; }

echo '[judge-only] === ensure python deps (pillow/openai/backoff only) ==='
python - <<'PY' || pip install pillow openai backoff
import backoff, openai, PIL  # noqa: F401
print("[judge-only] deps already present")
PY

echo '[judge-only] === source secrets ==='
source "$CREDS_FILE"
[[ -n "${OPENAI_API_KEY:-}" || -n "${OPENAI_API_BACKUP_KEY:-}" ]] || {
  echo "[judge-only][error] OPENAI_API_KEY not in $CREDS_FILE"; exit 1; }
# 直连 OPENAI_API_KEY 配额已耗尽(2026-07);有 gateway 凭据时优先走 gateway:
# eval 脚本的 --endpoint_target_uri 默认取 $OPENAI_GATEWAY_ENDPOINT,
# key 解析顺序 BACKUP 优先,把 gateway key 注入 BACKUP 即可全程走 gateway。
if [[ -n "${OPENAI_GATEWAY_API_KEY:-}" && -n "${OPENAI_GATEWAY_ENDPOINT:-}" ]]; then
  export OPENAI_API_BACKUP_KEY="$OPENAI_GATEWAY_API_KEY"
  echo "[judge-only] using phyagi gateway: $OPENAI_GATEWAY_ENDPOINT"
fi

mkdir -p "$JUDGE_OUTPUT_DIR"
cd "$REPO"
RC=0
python scripts/eval_with_original_om2w.py \
  --trajectories_dir "$TRAJECTORIES_DIR" \
  --output_path "$JUDGE_OUTPUT_DIR" \
  --tasks_file "$REPO/src/miniswewebagent/run/benchmarks/om2w_260220.json" \
  --model "$JUDGE_MODEL" \
  --score_threshold "$SCORE_THRESHOLD" \
  --task_level "$TASK_LEVEL" \
  --limit "$LIMIT" \
  --num_worker "$JUDGE_NUM_WORKERS" \
  --summary_path "$JUDGE_OUTPUT_DIR/summary.json" || RC=$?
echo "[judge-only] rc=$RC ; summary: $JUDGE_OUTPUT_DIR/summary.json"
exit "$RC"
