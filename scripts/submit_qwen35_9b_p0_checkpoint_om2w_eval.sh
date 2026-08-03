#!/usr/bin/env bash
# Submit a one-node P0 eval that converts and serves the completed Qwen3.5-9B
# PhiTrain checkpoint, then runs all 300 frozen last-observation OM2W tasks.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_qwen35_9b_p0_checkpoint_om2w_eval.sh [--dry-run]

Environment overrides:
  SOURCE_TRAINING_JOB, SOURCE_ROOT, EVAL_RUN_ID, JOB_NAME, MODEL_NAME,
  WORKERS, TP, GPU_MEMORY_UTILIZATION, FOLLOW_LOGS, CREDENTIALS_FILE,
  AIFSDK_ROOT.
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
PHITRAIN_SOURCE="${PHITRAIN_SOURCE:-$AIFSDK_ROOT/phitrain}"
SUBMIT="${SUBMIT:-$AIFSDK_ROOT/clusters/lambda/submission/submit_job.sh}"

SOURCE_TRAINING_JOB="${SOURCE_TRAINING_JOB:-luyadong-p0-abr-q35-9b-f0802-s294-f401f}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/pvc/experiments/luyadong/outputs/$SOURCE_TRAINING_JOB/train}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
EVAL_RUN_ID="${EVAL_RUN_ID:-qwen35-9b-s294-p0-lastobs-minimal-full-$(date -u +%Y%m%dT%H%M%SZ)}"

SOURCE_CONFIG_RUN="${SOURCE_CONFIG_RUN:-$MINI_WEB_AGENT_DIR/outputs/qwen36_27b_lastobs_minimal_20260802_204810_g0}"
CONFIG_SOURCE="${CONFIG_SOURCE:-$SOURCE_CONFIG_RUN/config_snapshot/merged_config.yaml}"
CONFIG_SPEC_SOURCE="${CONFIG_SPEC_SOURCE:-$SOURCE_CONFIG_RUN/config_snapshot/00_om2w_spb_vllm_lastobs_minimal.yaml}"
CONFIG_MANIFEST_SOURCE="${CONFIG_MANIFEST_SOURCE:-$SOURCE_CONFIG_RUN/config_snapshot/config_spec_manifest.json}"
TASKS_SOURCE="${TASKS_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
CHAT_TEMPLATE_SOURCE="${CHAT_TEMPLATE_SOURCE:-$PHITRAIN_SOURCE/scripts/tools/data/tokenization/tokenizers/Qwen3.5-no-auto-think/chat_template.jinja}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/home/luyadong/cred.sh}"

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
export JOB_NAME="${JOB_NAME:-${USER_ALIAS}-p0-absft-q35-9b-s294}"

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

for numeric_name in WORKERS JUDGE_NUM_PROC TP MAX_MODEL_LEN; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done

for required_path in \
    "$SUBMIT" \
    "$PHITRAIN_SOURCE" \
    "$CONFIG_SOURCE" \
    "$CONFIG_SPEC_SOURCE" \
    "$CONFIG_MANIFEST_SOURCE" \
    "$TASKS_SOURCE" \
    "$CHAT_TEMPLATE_SOURCE" \
    "$CREDENTIALS_FILE" \
    "$MINI_WEB_AGENT_DIR/scripts/run_qwen35_9b_p0_checkpoint_om2w_eval.sh"; do
    [[ -e "$required_path" ]] || {
        echo "[error] required path is missing: $required_path" >&2
        exit 1
    }
done

CONFIG_SHA256="$(sha256sum "$CONFIG_SOURCE" | awk '{print $1}')"
TASKS_SHA256="$(sha256sum "$TASKS_SOURCE" | awk '{print $1}')"
CHAT_TEMPLATE_SHA256="$(sha256sum "$CHAT_TEMPLATE_SOURCE" | awk '{print $1}')"
EXPECTED_CHAT_TEMPLATE_SHA256="e00b0a10f784841c4ee4c2dbd7e983244b59099f06920edf74d1f9804b71f035"
[[ "$CHAT_TEMPLATE_SHA256" == "$EXPECTED_CHAT_TEMPLATE_SHA256" ]] || {
    echo "[error] eval chat template does not match the tokenized training data" >&2
    echo "[error] expected=$EXPECTED_CHAT_TEMPLATE_SHA256 actual=$CHAT_TEMPLATE_SHA256" >&2
    exit 1
}
TASK_COUNT="$(python - "$TASKS_SOURCE" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"
[[ "$TASK_COUNT" == "300" ]] || {
    echo "[error] expected the complete 300-task OM2W file, found $TASK_COUNT" >&2
    exit 1
}

echo "[qwen35-submit] source_job=$SOURCE_TRAINING_JOB"
echo "[qwen35-submit] source_root=$SOURCE_ROOT"
echo "[qwen35-submit] run_id=$EVAL_RUN_ID tasks=$TASK_COUNT workers=$WORKERS"
echo "[qwen35-submit] node=1 gpus=8 tp=$TP priority=$PRIORITY class=$PRIORITY_CLASS_NAME"
echo "[qwen35-submit] workstream=$PROJECT_NAME job_base=$JOB_NAME"
echo "[qwen35-submit] config_sha256=$CONFIG_SHA256"
echo "[qwen35-submit] tasks_sha256=$TASKS_SHA256"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] validation passed; no Kubernetes resources were changed."
    exit 0
fi

SOURCE_PHASE="$(
    kubectl -n "$NAMESPACE" get job.batch.volcano.sh "$SOURCE_TRAINING_JOB" \
        -o jsonpath='{.status.state.phase}'
)"
[[ "$SOURCE_PHASE" == "Completed" ]] || {
    echo "[error] source training job must be Completed before eval submission" >&2
    echo "[error] job=$SOURCE_TRAINING_JOB phase=$SOURCE_PHASE" >&2
    exit 1
}

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-checkpoint-eval.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() {
    if [[ -n "${STAGING_PARENT:-}" &&
        -d "$STAGING_PARENT" &&
        "$(basename "$STAGING_PARENT")" == qwen35-checkpoint-eval.* ]]; then
        rm -rf -- "$STAGING_PARENT"
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
cp "$CHAT_TEMPLATE_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/qwen3_5_no_auto_think.jinja"

python - \
    "$UPLOAD_DIR/cluster_eval_assets/provenance.json" \
    "$SOURCE_TRAINING_JOB" \
    "$SOURCE_ROOT" \
    "$SOURCE_CONFIG_RUN" \
    "$CONFIG_SHA256" \
    "$TASKS_SHA256" \
    "$CHAT_TEMPLATE_SHA256" \
    "$TASK_COUNT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    source_training_job,
    source_root,
    source_config_run,
    config_sha256,
    tasks_sha256,
    template_sha256,
    task_count,
) = sys.argv[1:]
Path(output_path).write_text(
    json.dumps(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_training_job": source_training_job,
            "source_root": source_root,
            "source_config_run": source_config_run,
            "config_sha256": config_sha256,
            "tasks_sha256": tasks_sha256,
            "chat_template_sha256": template_sha256,
            "task_count": int(task_count),
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

EXTRA_ENV="SOURCE_TRAINING_JOB=$SOURCE_TRAINING_JOB,SOURCE_ROOT=$SOURCE_ROOT"
EXTRA_ENV+=",BASE_MODEL=$BASE_MODEL,EVAL_RUN_ID=$EVAL_RUN_ID,MODEL_NAME=$MODEL_NAME"
EXTRA_ENV+=",WORKERS=$WORKERS,JUDGE_NUM_PROC=$JUDGE_NUM_PROC,TP=$TP"
EXTRA_ENV+=",MAX_MODEL_LEN=$MAX_MODEL_LEN,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
EXTRA_ENV+=",CONFIG_SHA256=$CONFIG_SHA256,TASKS_SHA256=$TASKS_SHA256"
EXTRA_ENV+=",CHAT_TEMPLATE_SHA256=$CHAT_TEMPLATE_SHA256"
if [[ -n "${HF_TOKEN:-}" ]]; then
    EXTRA_ENV+=",HF_TOKEN=$HF_TOKEN,HF_HUB_TOKEN=$HF_TOKEN"
fi

FOLLOW_ARGS=()
[[ "$FOLLOW_LOGS" == "1" ]] && FOLLOW_ARGS=(--follow-logs)

bash "$SUBMIT" \
    --upload "$PHITRAIN_SOURCE" "$UPLOAD_DIR" \
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
    --cmd 'exec bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/scripts/run_qwen35_9b_p0_checkpoint_om2w_eval.sh'
