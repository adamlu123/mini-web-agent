#!/usr/bin/env bash
# Submit a one-node P0 Bonete evaluation that reproduces the frozen
# qwen36_27b_lastobs_minimal configuration.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_qwen36_27b_om2w_cluster_eval.sh [--dry-run]

Environment overrides:
  EVAL_RUN_ID, JOB_NAME, MODEL_ID, TASKS_SOURCE, WORKERS, TP,
  GPU_MEMORY_UTILIZATION, FOLLOW_LOGS, CREDENTIALS_FILE, AIFSDK_ROOT.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINI_WEB_AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AIFSDK_ROOT="${AIFSDK_ROOT:-/home/luyadong/sandbox/aifsdk}"
SUBMIT="${SUBMIT:-$AIFSDK_ROOT/clusters/lambda/submission/submit_job.sh}"

SOURCE_RUN="${SOURCE_RUN:-$MINI_WEB_AGENT_DIR/outputs/qwen36_27b_lastobs_minimal_20260802_204810_g0}"
CONFIG_SOURCE="${CONFIG_SOURCE:-$SOURCE_RUN/config_snapshot/merged_config.yaml}"
CONFIG_SPEC_SOURCE="${CONFIG_SPEC_SOURCE:-$SOURCE_RUN/config_snapshot/00_om2w_spb_vllm_lastobs_minimal.yaml}"
CONFIG_MANIFEST_SOURCE="${CONFIG_MANIFEST_SOURCE:-$SOURCE_RUN/config_snapshot/config_spec_manifest.json}"
# The local run split 300 tasks across g0 and g1. A one-node cluster run uses
# the complete task file referenced by the frozen config.
TASKS_SOURCE="${TASKS_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
CHAT_TEMPLATE_SOURCE="${CHAT_TEMPLATE_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/config/eval/qwen3_5_train_aligned.jinja}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/home/luyadong/cred.sh}"

MODEL_ID="${MODEL_ID:-/mnt/pvc/experiments/luyadong/models/webwright-teacher}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
# Empty keeps the frozen config's agent.step_limit; set to override it. It is
# woven into EVAL_RUN_ID / JOB_NAME so runs of different limits never collide.
STEP_LIMIT="${STEP_LIMIT:-}"
if [[ -n "$STEP_LIMIT" ]]; then
    [[ "$STEP_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
        echo "[error] STEP_LIMIT must be a positive integer: $STEP_LIMIT" >&2
        exit 2
    }
    STEP_LIMIT_TAG="-step${STEP_LIMIT}"
    # Volcano caps JOB_NAME at 40 chars, so the job tag is deliberately shorter
    # than the run-id tag.
    STEP_LIMIT_JOB_TAG="-sl${STEP_LIMIT}"
else
    STEP_LIMIT_TAG=""
    STEP_LIMIT_JOB_TAG=""
fi
EVAL_RUN_ID="${EVAL_RUN_ID:-qwen36-27b-lastobs-minimal-full${STEP_LIMIT_TAG}-$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

export USER_ALIAS="${USER_ALIAS:-${USER%@*}}"
export PROJECT_NAME="${PROJECT_NAME:-agenticbrain-sft}"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export NAMESPACE="${NAMESPACE:-bonete61}"
export JOB_NAME="${JOB_NAME:-${USER_ALIAS}-p0-absft-q36-27b-full${STEP_LIMIT_JOB_TAG}}"

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
    "$CONFIG_SOURCE" \
    "$CONFIG_SPEC_SOURCE" \
    "$CONFIG_MANIFEST_SOURCE" \
    "$TASKS_SOURCE" \
    "$CHAT_TEMPLATE_SOURCE" \
    "$CREDENTIALS_FILE" \
    "$MINI_WEB_AGENT_DIR/scripts/run_qwen36_27b_om2w_cluster_eval.sh"; do
    [[ -f "$required_file" ]] || {
        echo "[error] required file is missing: $required_file" >&2
        exit 1
    }
done

CONFIG_SHA256="$(sha256sum "$CONFIG_SOURCE" | awk '{print $1}')"
TASKS_SHA256="$(sha256sum "$TASKS_SOURCE" | awk '{print $1}')"
CHAT_TEMPLATE_SHA256="$(sha256sum "$CHAT_TEMPLATE_SOURCE" | awk '{print $1}')"
TASK_COUNT="$(python - "$TASKS_SOURCE" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"

echo "[qwen36-submit] model=$MODEL_ID run_id=$EVAL_RUN_ID"
echo "[qwen36-submit] tasks=$TASK_COUNT workers=$WORKERS config=$CONFIG_SOURCE"
echo "[qwen36-submit] node=1 gpus=8 tp=$TP priority=$PRIORITY class=$PRIORITY_CLASS_NAME"
echo "[qwen36-submit] workstream=$PROJECT_NAME job_base=$JOB_NAME"
echo "[qwen36-submit] config_sha256=$CONFIG_SHA256"
echo "[qwen36-submit] tasks_sha256=$TASKS_SHA256"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] validation passed; no Kubernetes resources were changed."
    exit 0
fi

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/qwen36-cluster-eval.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() {
    if [[ -n "${STAGING_PARENT:-}" &&
          -d "$STAGING_PARENT" &&
          "$(basename "$STAGING_PARENT")" == qwen36-cluster-eval.* ]]; then
        rm -rf "$STAGING_PARENT"
    fi
}
trap cleanup_staging EXIT

mkdir -p "$UPLOAD_DIR/cluster_eval_assets"
for relpath in src scripts agent_runtime om2w_judge; do
    cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
for relpath in pyproject.toml README.md; do
    cp -a "$MINI_WEB_AGENT_DIR/$relpath" "$UPLOAD_DIR/"
done
cp "$CONFIG_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/merged_config.yaml"
cp "$CONFIG_SPEC_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/source_config.yaml"
cp "$CONFIG_MANIFEST_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/config_spec_manifest.json"
cp "$TASKS_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/tasks.json"
cp "$CHAT_TEMPLATE_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/qwen3_5_train_aligned.jinja"

python - \
    "$UPLOAD_DIR/cluster_eval_assets/provenance.json" \
    "$SOURCE_RUN" \
    "$CONFIG_SHA256" \
    "$TASKS_SHA256" \
    "$CHAT_TEMPLATE_SHA256" \
    "$TASK_COUNT" \
    "$MODEL_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    source_run,
    config_sha256,
    tasks_sha256,
    template_sha256,
    task_count,
    model_id,
) = sys.argv[1:]
Path(output_path).write_text(
    json.dumps(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": source_run,
            "config_sha256": config_sha256,
            "tasks_sha256": tasks_sha256,
            "chat_template_sha256": template_sha256,
            "task_count": int(task_count),
            "model_id": model_id,
            "node_count": 1,
            "gpu_count": 8,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-${USER_ALIAS}-webchain-sampling-creds}"
kubectl -n "$NAMESPACE" create secret generic "$CREDENTIALS_SECRET" \
    --from-file=cred.sh="$CREDENTIALS_FILE" \
    --dry-run=client -o yaml | kubectl -n "$NAMESPACE" apply -f -

EXTRA_ENV="EVAL_RUN_ID=$EVAL_RUN_ID,MODEL_ID=$MODEL_ID,MODEL_NAME=$MODEL_NAME"
EXTRA_ENV+=",WORKERS=$WORKERS,JUDGE_NUM_PROC=$JUDGE_NUM_PROC,TP=$TP"
EXTRA_ENV+=",MAX_MODEL_LEN=$MAX_MODEL_LEN,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
EXTRA_ENV+=",CONFIG_SHA256=$CONFIG_SHA256,TASKS_SHA256=$TASKS_SHA256"
EXTRA_ENV+=",CHAT_TEMPLATE_SHA256=$CHAT_TEMPLATE_SHA256"
[[ -n "$STEP_LIMIT" ]] && EXTRA_ENV+=",STEP_LIMIT=$STEP_LIMIT"
[[ -n "${MAX_CONTEXT_TOKENS:-}" ]] && EXTRA_ENV+=",MAX_CONTEXT_TOKENS=$MAX_CONTEXT_TOKENS"
if [[ -n "${HF_TOKEN:-}" ]]; then
    EXTRA_ENV+=",HF_TOKEN=$HF_TOKEN,HF_HUB_TOKEN=$HF_TOKEN"
fi

FOLLOW_ARGS=()
[[ "$FOLLOW_LOGS" == "1" ]] && FOLLOW_ARGS=(--follow-logs)

bash "$SUBMIT" \
    --upload "$UPLOAD_DIR" \
    --image "${IMAGE:-aifrontiers.azurecr.io/nvidia-26.06-pytorch-2.12.1-torchao-0.17.0-te-2.16.1-deepspeed-0.19.2-fa2-1f7ce2f-fa4-4.0.0b19-vllm-0.24.0:20260707}" \
    --acr \
    --node 1 \
    --gpu-per-node 8 \
    --cpu "${EVAL_CPU:-64}" \
    --memory "${EVAL_MEMORY:-512Gi}" \
    --shm "${EVAL_SHM:-64Gi}" \
    --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" \
    --extra-env-vars "$EXTRA_ENV" \
    "${FOLLOW_ARGS[@]}" \
    --cmd 'exec bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/scripts/run_qwen36_27b_om2w_cluster_eval.sh'
