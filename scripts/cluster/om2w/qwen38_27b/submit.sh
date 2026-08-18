#!/usr/bin/env bash
# Submit a one-node P0 Bonete evaluation of stock Qwen3.8-27B on the SPB
# persistent-browser OM2W benchmark.
#
# Uses eval/om2w_spb_vllm_lastobs.yaml as the base config -- prompts, observation
# truncation, history construction (last_obs), step limit and judge -- and runs it
# unmodified: no policy overlay is applied by default.
#
# Per-run tweaks go through EXTRA_CFG (dotted `-c` overrides, applied last), or
# OVERLAY_SOURCE=<yaml> to layer a whole file, e.g.
#   OVERLAY_SOURCE=$PWD/src/miniswewebagent/config/eval/om2w_spb_vllm_qwen38_27b.yaml
#
# CONFIG_SOURCE is overridable, so a frozen run snapshot can still be replayed
# byte-for-byte when a result must stay comparable with an older baseline:
#   SNAP=outputs/runs/qwen36_27b_lastobs_minimal_20260802_204810/g0/config_snapshot
#   CONFIG_SOURCE=$SNAP/merged_config.yaml \
#   CONFIG_SPEC_SOURCE=$SNAP/00_om2w_spb_vllm_lastobs_minimal.yaml \
#   CONFIG_MANIFEST_SOURCE=$SNAP/config_spec_manifest.json \
#     scripts/cluster/om2w/qwen38_27b/submit.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cluster/om2w/qwen38_27b/submit.sh [--dry-run]

Environment overrides:
  EVAL_RUN_ID, JOB_NAME, MODEL_ID, MODEL_NAME, TASKS_SOURCE, WORKERS, TP,
  MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, JUDGE_NUM_PROC, STEP_LIMIT, EXTRA_CFG,
  CHAT_TEMPLATE, IMAGE, FOLLOW_LOGS, CREDENTIALS_FILE, AIFSDK_ROOT,
  CONFIG_SOURCE, CONFIG_SPEC_SOURCE, CONFIG_MANIFEST_SOURCE, SOURCE_RUN,
  OVERLAY_SOURCE (empty by default -- no overlay is applied),
  GOLD_SOURCE (task-matched gold trajectories to stage alongside the run),
  PROJECT_NAME (any agenticbrain* workstream), PRIORITY (p0..p3),
  PRIORITY_CLASS_NAME (high|medium|low; defaults from PRIORITY).
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
# shellcheck source=../../../lib/cluster_submit.sh
source "$SCRIPT_DIR/../../../lib/cluster_submit.sh"
MINI_WEB_AGENT_DIR="$(mwa_cluster_repo_root "$SCRIPT_DIR")"
AIFSDK_ROOT="${AIFSDK_ROOT:-/home/luyadong/sandbox/aifsdk}"
SUBMIT="${SUBMIT:-$AIFSDK_ROOT/clusters/lambda/submission/submit_job.sh}"

# Empty unless a frozen run snapshot is being replayed; recorded in provenance.
SOURCE_RUN="${SOURCE_RUN:-}"
CONFIG_SOURCE="${CONFIG_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/config/eval/om2w_spb_vllm_lastobs.yaml}"
# The base config is a plain spec, not a merged snapshot, so it is its own
# provenance copy and there is no config_spec_manifest.json to stage.
CONFIG_SPEC_SOURCE="${CONFIG_SPEC_SOURCE:-$CONFIG_SOURCE}"
CONFIG_MANIFEST_SOURCE="${CONFIG_MANIFEST_SOURCE:-}"
# Empty by default: the base config is used as-is.
OVERLAY_SOURCE="${OVERLAY_SOURCE:-}"
TASKS_SOURCE="${TASKS_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
# Required by eval/om2w_spb_vllm_lastobs_gold_guided.yaml, ignored otherwise. Only
# task.json / result.json / plan.md are staged per task: those are all the guide
# renderer reads, while a full trajectory export is gigabytes of screenshots.
# run.sh points run.gold_trajectory_dir at the staged copy when it is present.
GOLD_SOURCE="${GOLD_SOURCE:-}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/home/luyadong/cred.sh}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_NAME="${MODEL_NAME:-qwen38_27b}"
# Empty keeps the model's own chat template, which is correct for a stock
# instruct checkpoint. Woven into no name, since it is rarely set.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
# Empty keeps the base config's agent.step_limit (50). Woven into
# EVAL_RUN_ID / JOB_NAME so runs of different limits never collide.
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
EVAL_RUN_ID="${EVAL_RUN_ID:-qwen38-27b-lastobs-full${STEP_LIMIT_TAG}-$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
EXTRA_CFG="${EXTRA_CFG:-}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

export USER_ALIAS="${USER_ALIAS:-${USER%@*}}"
export PROJECT_NAME="${PROJECT_NAME:-agenticbrain-sft}"
export PRIORITY="${PRIORITY:-p0}"
# Scheduling class defaults to the workstream priority: p0 preempts, p1 is the
# normal shared-cluster citizen, p2/p3 are best effort. Override explicitly to
# push a p1 up to `high`, which is also common in this namespace.
case "$PRIORITY" in
    p0) DEFAULT_PRIORITY_CLASS_NAME=high ;;
    p1) DEFAULT_PRIORITY_CLASS_NAME=medium ;;
    p2|p3) DEFAULT_PRIORITY_CLASS_NAME=low ;;
    *)
        echo "[error] PRIORITY must be one of p0, p1, p2, p3: $PRIORITY" >&2
        exit 2
        ;;
esac
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-$DEFAULT_PRIORITY_CLASS_NAME}"
export NAMESPACE="${NAMESPACE:-bonete61}"
# Priority is woven into the job name so a p0 and a p1 of the same eval never
# collide on one Volcano job.
export JOB_NAME="${JOB_NAME:-${USER_ALIAS}-${PRIORITY}-absft-q38-27b-full${STEP_LIMIT_JOB_TAG}}"

case "$PRIORITY_CLASS_NAME" in
    high|medium|low) ;;
    *)
        echo "[error] PRIORITY_CLASS_NAME must be one of high, medium, low: $PRIORITY_CLASS_NAME" >&2
        exit 2
        ;;
esac
[[ "$PROJECT_NAME" == agenticbrain* ]] || {
    echo "[error] this launcher requires an agenticbrain workstream: $PROJECT_NAME" >&2
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
    ${CONFIG_MANIFEST_SOURCE:+"$CONFIG_MANIFEST_SOURCE"} \
    ${OVERLAY_SOURCE:+"$OVERLAY_SOURCE"} \
    "$TASKS_SOURCE" \
    "$CREDENTIALS_FILE" \
    "$SCRIPT_DIR/run.sh"; do
    [[ -f "$required_file" ]] || {
        echo "[error] required file is missing: $required_file" >&2
        exit 1
    }
done
if [[ -n "$CHAT_TEMPLATE" && ! -f "$CHAT_TEMPLATE" ]]; then
    echo "[error] chat template is missing: $CHAT_TEMPLATE" >&2
    exit 1
fi
if [[ -n "$GOLD_SOURCE" && ! -d "$GOLD_SOURCE" ]]; then
    echo "[error] gold trajectory directory is missing: $GOLD_SOURCE" >&2
    exit 1
fi

CONFIG_SHA256="$(mwa_sha256 "$CONFIG_SOURCE")"
OVERLAY_SHA256=""
[[ -n "$OVERLAY_SOURCE" ]] && OVERLAY_SHA256="$(mwa_sha256 "$OVERLAY_SOURCE")"
TASKS_SHA256="$(mwa_sha256 "$TASKS_SOURCE")"
TASK_COUNT="$(mwa_json_array_length "$TASKS_SOURCE")"

echo "[qwen38-submit] model=$MODEL_ID run_id=$EVAL_RUN_ID"
echo "[qwen38-submit] tasks=$TASK_COUNT workers=$WORKERS config=$CONFIG_SOURCE"
echo "[qwen38-submit] overlay=${OVERLAY_SOURCE:-<none>}"
echo "[qwen38-submit] gold=${GOLD_SOURCE:-<none>}"
echo "[qwen38-submit] node=1 gpus=8 tp=$TP max_model_len=$MAX_MODEL_LEN"
echo "[qwen38-submit] priority=$PRIORITY class=$PRIORITY_CLASS_NAME"
echo "[qwen38-submit] workstream=$PROJECT_NAME job_base=$JOB_NAME"
echo "[qwen38-submit] config_sha256=$CONFIG_SHA256"
[[ -n "$OVERLAY_SHA256" ]] && echo "[qwen38-submit] overlay_sha256=$OVERLAY_SHA256"
echo "[qwen38-submit] tasks_sha256=$TASKS_SHA256"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] validation passed; no Kubernetes resources were changed."
    exit 0
fi

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-cluster-eval.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
cleanup_staging() {
    if [[ -n "${STAGING_PARENT:-}" &&
          -d "$STAGING_PARENT" &&
          "$(basename "$STAGING_PARENT")" == qwen38-cluster-eval.* ]]; then
        rm -rf "$STAGING_PARENT"
    fi
}
trap cleanup_staging EXIT

mkdir -p "$UPLOAD_DIR/cluster_eval_assets"
mwa_stage_cluster_repo "$MINI_WEB_AGENT_DIR" "$UPLOAD_DIR"
cp "$CONFIG_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/merged_config.yaml"
cp "$CONFIG_SPEC_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/source_config.yaml"
if [[ -n "$CONFIG_MANIFEST_SOURCE" ]]; then
    cp "$CONFIG_MANIFEST_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/config_spec_manifest.json"
fi
if [[ -n "$OVERLAY_SOURCE" ]]; then
    cp "$OVERLAY_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/overlay_config.yaml"
fi
cp "$TASKS_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/tasks.json"
if [[ -n "$GOLD_SOURCE" ]]; then
    python - "$GOLD_SOURCE" "$UPLOAD_DIR/cluster_eval_assets/gold" "$TASKS_SOURCE" <<'GOLDPY'
import json
import shutil
import sys
from pathlib import Path

source, destination, tasks_path = (Path(value) for value in sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
wanted = {
    str(task.get("task_id") or "")
    for task in json.loads(tasks_path.read_text(encoding="utf-8"))
}
wanted.discard("")

staged = 0
missing = []
for task_id in sorted(wanted):
    task_dir = source / task_id
    # The renderer hard-fails per task on either file, so a gap is a submission
    # error rather than something to discover 300 running tasks later.
    if not ((task_dir / "task.json").is_file() and (task_dir / "result.json").is_file()):
        missing.append(task_id)
        continue
    target = destination / task_id
    target.mkdir(parents=True, exist_ok=True)
    for name in ("task.json", "result.json", "plan.md"):
        if (task_dir / name).is_file():
            shutil.copy2(task_dir / name, target / name)
    staged += 1

if missing:
    raise SystemExit(
        f"[error] {len(missing)} of {len(wanted)} tasks have no gold trajectory "
        f"under {source}: {', '.join(missing[:5])}"
    )
print(f"[qwen38-submit] gold_tasks_staged={staged}")
GOLDPY
fi

python - \
    "$UPLOAD_DIR/cluster_eval_assets/provenance.json" \
    "$SOURCE_RUN" \
    "${CONFIG_SOURCE#"$MINI_WEB_AGENT_DIR/"}" \
    "$CONFIG_SHA256" \
    "$OVERLAY_SHA256" \
    "$TASKS_SHA256" \
    "$TASK_COUNT" \
    "$MODEL_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    source_run,
    config_source,
    config_sha256,
    overlay_sha256,
    tasks_sha256,
    task_count,
    model_id,
) = sys.argv[1:]
Path(output_path).write_text(
    json.dumps(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": source_run or None,
            "config_source": config_source,
            "config_sha256": config_sha256,
            "overlay_sha256": overlay_sha256 or None,
            "tasks_sha256": tasks_sha256,
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
mwa_apply_credentials_secret "$NAMESPACE" "$CREDENTIALS_SECRET" "$CREDENTIALS_FILE"

EXTRA_ENV="EVAL_RUN_ID=$EVAL_RUN_ID,MODEL_ID=$MODEL_ID,MODEL_NAME=$MODEL_NAME"
EXTRA_ENV+=",WORKERS=$WORKERS,JUDGE_NUM_PROC=$JUDGE_NUM_PROC,TP=$TP"
EXTRA_ENV+=",MAX_MODEL_LEN=$MAX_MODEL_LEN,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
EXTRA_ENV+=",CONFIG_SHA256=$CONFIG_SHA256"
[[ -n "$OVERLAY_SHA256" ]] && EXTRA_ENV+=",OVERLAY_SHA256=$OVERLAY_SHA256"
EXTRA_ENV+=",TASKS_SHA256=$TASKS_SHA256"
[[ -n "$CHAT_TEMPLATE" ]] && EXTRA_ENV+=",CHAT_TEMPLATE=$CHAT_TEMPLATE"
[[ -n "$STEP_LIMIT" ]] && EXTRA_CFG="agent.step_limit=$STEP_LIMIT${EXTRA_CFG:+ $EXTRA_CFG}"
[[ -n "$EXTRA_CFG" ]] && EXTRA_ENV+=",EXTRA_CFG=$EXTRA_CFG"
if [[ -n "${HF_TOKEN:-}" ]]; then
    EXTRA_ENV+=",HF_TOKEN=$HF_TOKEN,HF_HUB_TOKEN=$HF_TOKEN"
fi

FOLLOW_ARGS=()
[[ "$FOLLOW_LOGS" == "1" ]] && FOLLOW_ARGS=(--follow-logs)

bash "$SUBMIT" \
    --upload "$UPLOAD_DIR" \
    --image "${IMAGE:-aifrontiers.azurecr.io/nvidia-26.06-pytorch-2.13.0-torchao-0.17.0-te-2.17-deepspeed-0.19.2-fa2-77aacb6-fa4-4.0.0b19-vllm-0.25.1:20260717}" \
    --acr \
    --node 1 \
    --gpu-per-node 8 \
    --cpu "${EVAL_CPU:-64}" \
    --memory "${EVAL_MEMORY:-512Gi}" \
    --shm "${EVAL_SHM:-64Gi}" \
    --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" \
    --extra-env-vars "$EXTRA_ENV" \
    "${FOLLOW_ARGS[@]}" \
    --cmd 'exec bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/scripts/cluster/om2w/qwen38_27b/run.sh'
