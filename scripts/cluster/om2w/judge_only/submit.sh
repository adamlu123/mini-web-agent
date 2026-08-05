#!/usr/bin/env bash
# Submit a CPU-only P0 job that re-runs just the OM2W judge over an eval run
# whose trajectories already sit on the PVC.
#
#   RUN_ROOT=/mnt/pvc/experiments/luyadong/evals/<run-id> \
#   JOB_SUFFIX=q36-27b \
#     bash scripts/submit_om2w_judge_only_cluster.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cluster/om2w/judge_only/submit.sh [--dry-run]

Required environment:
  RUN_ROOT     PVC path of the eval run to re-score (must contain outputs/).

Environment overrides:
  JOB_SUFFIX, JOB_NAME, TASKS_SOURCE, EVAL_DIR_NAME, JUDGE_MODEL,
  JUDGE_NUM_PROC, JUDGE_ENDPOINT, EVAL_CPU, EVAL_MEMORY, FOLLOW_LOGS,
  CREDENTIALS_FILE, AIFSDK_ROOT.
EOF
}

DRY_RUN="${DRY_RUN:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

: "${RUN_ROOT:?RUN_ROOT is not set (PVC path of the eval run to re-score)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../lib/cluster_submit.sh
source "$SCRIPT_DIR/../../../lib/cluster_submit.sh"
MINI_WEB_AGENT_DIR="$(mwa_cluster_repo_root "$SCRIPT_DIR")"
AIFSDK_ROOT="${AIFSDK_ROOT:-/home/luyadong/sandbox/aifsdk}"
SUBMIT="${SUBMIT:-$AIFSDK_ROOT/clusters/lambda/submission/submit_job.sh}"

TASKS_SOURCE="${TASKS_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/home/luyadong/cred.sh}"

EVAL_DIR_NAME="${EVAL_DIR_NAME:-outputs_eval_judgeonly_1}"
JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://gateway.phyagi.net/api/responses}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

export USER_ALIAS="${USER_ALIAS:-${USER%@*}}"
export PROJECT_NAME="${PROJECT_NAME:-agenticbrain-sft}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export NAMESPACE="${NAMESPACE:-bonete61}"
JOB_SUFFIX="${JOB_SUFFIX:-judge}"
export JOB_NAME="${JOB_NAME:-${USER_ALIAS}-p0-judge-${JOB_SUFFIX}}"

[[ "$PRIORITY" == "p0" && "$PRIORITY_CLASS_NAME" == "high" ]] || {
    echo "[error] this launcher requires PRIORITY=p0 and PRIORITY_CLASS_NAME=high" >&2
    exit 2
}
[[ "$PROJECT_NAME" == "agenticbrain-sft" ]] || {
    echo "[error] this launcher requires PROJECT_NAME=agenticbrain-sft" >&2
    exit 2
}
[[ "$FOLLOW_LOGS" == "0" || "$FOLLOW_LOGS" == "1" ]] || {
    echo "[error] FOLLOW_LOGS must be 0 or 1" >&2
    exit 2
}
if (( ${#JOB_NAME} > 40 )); then
    echo "[error] JOB_NAME must be at most 40 characters for Volcano pod names: $JOB_NAME" >&2
    exit 2
fi

for required_file in \
    "$SUBMIT" \
    "$TASKS_SOURCE" \
    "$CREDENTIALS_FILE" \
    "$SCRIPT_DIR/run.sh"; do
    [[ -f "$required_file" ]] || {
        echo "[error] required file is missing: $required_file" >&2
        exit 1
    }
done

echo "[judge-submit] run_root=$RUN_ROOT"
echo "[judge-submit] job=$JOB_NAME eval_dir=$EVAL_DIR_NAME"
echo "[judge-submit] judge=$JUDGE_MODEL num_proc=$JUDGE_NUM_PROC endpoint=$JUDGE_ENDPOINT"
echo "[judge-submit] node=1 gpus=0 (CPU only) priority=$PRIORITY class=$PRIORITY_CLASS_NAME"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] validation passed; no Kubernetes resources were changed."
    exit 0
fi

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/om2w-judge-only.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() {
    if [[ -n "${STAGING_PARENT:-}" &&
          -d "$STAGING_PARENT" &&
          "$(basename "$STAGING_PARENT")" == om2w-judge-only.* ]]; then
        rm -rf "$STAGING_PARENT"
    fi
}
trap cleanup_staging EXIT

mkdir -p "$UPLOAD_DIR/cluster_eval_assets"
mwa_stage_cluster_repo "$MINI_WEB_AGENT_DIR" "$UPLOAD_DIR"
cp "$TASKS_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/tasks.json"

CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-${USER_ALIAS}-webchain-sampling-creds}"
mwa_apply_credentials_secret "$NAMESPACE" "$CREDENTIALS_SECRET" "$CREDENTIALS_FILE"

EXTRA_ENV="RUN_ROOT=$RUN_ROOT,EVAL_DIR_NAME=$EVAL_DIR_NAME"
EXTRA_ENV+=",JUDGE_MODEL=$JUDGE_MODEL,JUDGE_NUM_PROC=$JUDGE_NUM_PROC"
EXTRA_ENV+=",JUDGE_ENDPOINT=$JUDGE_ENDPOINT"

FOLLOW_ARGS=()
[[ "$FOLLOW_LOGS" == "1" ]] && FOLLOW_ARGS=(--follow-logs)

bash "$SUBMIT" \
    --upload "$UPLOAD_DIR" \
    --image "${IMAGE:-aifrontiers.azurecr.io/nvidia-26.06-pytorch-2.12.1-torchao-0.17.0-te-2.16.1-deepspeed-0.19.2-fa2-1f7ce2f-fa4-4.0.0b19-vllm-0.24.0:20260707}" \
    --acr \
    --node 1 \
    --gpu-per-node 0 \
    --cpu "${EVAL_CPU:-32}" \
    --memory "${EVAL_MEMORY:-128Gi}" \
    --shm "${EVAL_SHM:-16Gi}" \
    --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" \
    --extra-env-vars "$EXTRA_ENV" \
    "${FOLLOW_ARGS[@]}" \
    --cmd 'exec bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/scripts/cluster/om2w/judge_only/run.sh'
