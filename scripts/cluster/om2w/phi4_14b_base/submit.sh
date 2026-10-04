#!/usr/bin/env bash
# Submit a one-node Bonete evaluation of the Phi-4-14B *base* checkpoint on the
# SPB persistent-browser OM2W benchmark.
#
# The untrained baseline for scripts/cluster/om2w/phi4_14b/submit.sh: same eval
# config (eval/om2w_spb_vllm_lastobs.yaml, unmodified, no overlay by default),
# same tasks, same judge, same 1-node 8-GPU TP=2 x DP=4 serving shape. The only
# difference is the policy -- BASE_CKPT is the checkpoint the webwright SFT run
# was initialised from, i.e. the exact model phi4_14b/submit.sh passes as
# BASE_MODEL to supply the tokenizer and chat template. Scoring the two against
# each other is what isolates the effect of that SFT.
#
# BASE_CKPT is a complete Hugging Face directory, so unlike the SFT launcher
# there is no export to assemble and nothing is written next to the weights: it
# is served in place, read-only.
#
# Per-run tweaks go through EXTRA_CFG (dotted `-c` overrides, applied last), or
# OVERLAY_SOURCE=<yaml> to layer a whole file.
#
#   PRIORITY=p1 PROJECT_NAME=agenticbrain \
#     bash scripts/cluster/om2w/phi4_14b_base/submit.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cluster/om2w/phi4_14b_base/submit.sh [--dry-run]

Environment overrides:
  BASE_CKPT, EVAL_RUN_ID, JOB_NAME, MODEL_NAME, TASKS_SOURCE, WORKERS, TP, DP,
  DTYPE, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, JUDGE_NUM_PROC, STEP_LIMIT,
  MAX_OUTPUT_TOKENS, CONTEXT_MARGIN, MAX_CONTEXT_TOKENS, EXTRA_CFG, IMAGE,
  FOLLOW_LOGS, CREDENTIALS_FILE, AIFSDK_ROOT, CONFIG_SOURCE, CONFIG_SPEC_SOURCE,
  CONFIG_MANIFEST_SOURCE,
  OVERLAY_SOURCE (empty by default -- no overlay is applied),
  GOLD_SOURCE (task-matched gold trajectories to stage alongside the run),
  PROJECT_NAME (any agenticbrain* workstream), PRIORITY (p0..p3),
  PRIORITY_CLASS_NAME (high|medium|low; defaults from PRIORITY),
  BORROW_WS / BORROW_NGPU (lender workstream and GPUs to put on its quota).
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

# The base policy: a complete Hugging Face directory on the PVC, served as is.
# This is exactly the BASE_MODEL that scripts/cluster/om2w/phi4_14b/submit.sh
# borrows the tokenizer and chat template from, so the SFT run and this baseline
# share a tokenizer, a prompt format and a 32768-token window by construction.
BASE_CKPT="${BASE_CKPT:-/mnt/pvc/datasets/rlscaling/agoswami/ckpts_intermediate_01/agoswami-p0-rlscaling-phi4-resume-9b184/6000/llama}"
MODEL_NAME="${MODEL_NAME:-base_ckpt}"

CONFIG_SOURCE="${CONFIG_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/config/eval/om2w_spb_vllm_lastobs.yaml}"
# The base config is a plain spec, not a merged snapshot, so it is its own
# provenance copy and there is no config_spec_manifest.json to stage.
CONFIG_SPEC_SOURCE="${CONFIG_SPEC_SOURCE:-$CONFIG_SOURCE}"
CONFIG_MANIFEST_SOURCE="${CONFIG_MANIFEST_SOURCE:-}"
# Empty by default: the base config is used as-is.
OVERLAY_SOURCE="${OVERLAY_SOURCE:-}"
TASKS_SOURCE="${TASKS_SOURCE:-$MINI_WEB_AGENT_DIR/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
GOLD_SOURCE="${GOLD_SOURCE:-}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/home/luyadong/cred.sh}"

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
EVAL_RUN_ID="${EVAL_RUN_ID:-phi4-14b-base-lastobs-full${STEP_LIMIT_TAG}-$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
# num_key_value_heads=10, so vLLM accepts only TP in {1,2,5} on an 8-GPU node --
# the rest of the node is filled with data parallelism. TP*DP must be <= 8.
TP="${TP:-2}"
DP="${DP:-4}"
# config.json max_position_embeddings for this checkpoint.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# This checkpoint is stored float32 and vLLM's `auto` would resolve that to
# fp16, whose range does not cover bf16-trained Phi-4 activations. See run.sh.
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
EXTRA_CFG="${EXTRA_CFG:-}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

export USER_ALIAS="${USER_ALIAS:-${USER%@*}}"
export PROJECT_NAME="${PROJECT_NAME:-agenticbrain}"
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
export JOB_NAME="${JOB_NAME:-${USER_ALIAS}-${PRIORITY}-abbase-phi4-14b${STEP_LIMIT_JOB_TAG}}"

GPU_COUNT=8
BORROW_WS="${BORROW_WS:-}"
BORROW_NGPU="${BORROW_NGPU:-$GPU_COUNT}"

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
for numeric_name in WORKERS JUDGE_NUM_PROC TP DP MAX_MODEL_LEN; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done
if (( TP * DP > GPU_COUNT )); then
    echo "[error] TP*DP=$((TP * DP)) exceeds the job's $GPU_COUNT GPUs" >&2
    exit 2
fi
# vLLM shards KV heads across TP ranks and needs total_num_kv_heads % tp == 0
# (or tp % total_num_kv_heads == 0). This checkpoint has 10, so TP=8 would fail
# at engine start, an hour into a p0 allocation.
PHI4_KV_HEADS="${PHI4_KV_HEADS:-10}"
if (( PHI4_KV_HEADS % TP != 0 && TP % PHI4_KV_HEADS != 0 )); then
    echo "[error] TP=$TP is incompatible with num_key_value_heads=$PHI4_KV_HEADS" >&2
    echo "[error] pick a TP that divides it (1, 2, 5) or a multiple of it" >&2
    exit 2
fi
case "$DTYPE" in
    bfloat16|float16|float32|half|auto) ;;
    *)
        echo "[error] DTYPE must be one of bfloat16, float16, float32, half, auto: $DTYPE" >&2
        exit 2
        ;;
esac

# An invalid borrow is ignored rather than rejected, so the lender/borrower pair
# is validated here against the policy the deployed collector actually reads.
# Reference: .github/skills/submit-cluster-job/references/quota-and-reaping.md
if [[ -n "$BORROW_WS" ]]; then
    [[ "$BORROW_NGPU" =~ ^[1-9][0-9]*$ ]] || {
        echo "[error] BORROW_NGPU must be a positive integer: $BORROW_NGPU" >&2
        exit 2
    }
    (( BORROW_NGPU <= GPU_COUNT )) || {
        echo "[error] BORROW_NGPU exceeds the job's $GPU_COUNT physical GPUs: $BORROW_NGPU" >&2
        exit 2
    }
    if [[ "$FOLLOW_LOGS" == "1" ]]; then
        echo "[error] FOLLOW_LOGS=1 blocks until the job ends, so the borrow labels would be applied far too late" >&2
        exit 2
    fi
    COLLECTOR_REF="${COLLECTOR_REF:-origin/main}"
    BORROW_LIMITS="$(git -C "$AIFSDK_ROOT" show "$COLLECTOR_REF:clusters/lambda/monitor/collector/borrow-limit.json")"
    WORKSTREAM_ALIASES="$(git -C "$AIFSDK_ROOT" show "$COLLECTOR_REF:clusters/lambda/monitor/collector/workstream-aliases.json")"
    python - "$BORROW_WS" "$PROJECT_NAME" "$BORROW_NGPU" "$BORROW_LIMITS" "$WORKSTREAM_ALIASES" <<'PY'
import json
import sys

lender, borrower, requested, limits_text, aliases_text = sys.argv[1:]
aliases = json.loads(aliases_text)
lender = aliases.get(lender, lender)
borrower = aliases.get(borrower, borrower)
if lender == borrower:
    raise SystemExit(f"[error] borrow-ws resolves to the job's own workstream: {lender}")
cap = json.loads(limits_text).get(lender, {}).get(borrower)
if cap is None:
    raise SystemExit(f"[error] borrow pair {lender} -> {borrower} is not in borrow-limit.json")
if int(requested) > int(cap):
    raise SystemExit(f"[error] borrow of {requested} GPUs exceeds the {lender} -> {borrower} cap of {cap}")
print(f"[phi4-base-submit] borrow validated: {lender} -> {borrower} {requested}/{cap} GPUs")
PY
fi

for required_path in \
    "$SUBMIT" \
    "$CONFIG_SOURCE" \
    "$CONFIG_SPEC_SOURCE" \
    ${CONFIG_MANIFEST_SOURCE:+"$CONFIG_MANIFEST_SOURCE"} \
    ${OVERLAY_SOURCE:+"$OVERLAY_SOURCE"} \
    "$TASKS_SOURCE" \
    "$CREDENTIALS_FILE" \
    "$SCRIPT_DIR/run.sh"; do
    [[ -e "$required_path" ]] || {
        echo "[error] required path is missing: $required_path" >&2
        exit 1
    }
done
if [[ -n "$GOLD_SOURCE" && ! -d "$GOLD_SOURCE" ]]; then
    echo "[error] gold trajectory directory is missing: $GOLD_SOURCE" >&2
    exit 1
fi

CONFIG_SHA256="$(mwa_sha256 "$CONFIG_SOURCE")"
OVERLAY_SHA256=""
[[ -n "$OVERLAY_SOURCE" ]] && OVERLAY_SHA256="$(mwa_sha256 "$OVERLAY_SOURCE")"
TASKS_SHA256="$(mwa_sha256 "$TASKS_SOURCE")"
TASK_COUNT="$(mwa_json_array_length "$TASKS_SOURCE")"

echo "[phi4-base-submit] base_ckpt=$BASE_CKPT"
echo "[phi4-base-submit] run_id=$EVAL_RUN_ID model_name=$MODEL_NAME dtype=$DTYPE"
echo "[phi4-base-submit] tasks=$TASK_COUNT workers=$WORKERS config=$CONFIG_SOURCE"
echo "[phi4-base-submit] overlay=${OVERLAY_SOURCE:-<none>}"
echo "[phi4-base-submit] gold=${GOLD_SOURCE:-<none>}"
echo "[phi4-base-submit] node=1 gpus=$GPU_COUNT tp=$TP dp=$DP max_model_len=$MAX_MODEL_LEN"
echo "[phi4-base-submit] priority=$PRIORITY class=$PRIORITY_CLASS_NAME"
echo "[phi4-base-submit] workstream=$PROJECT_NAME job_base=$JOB_NAME"
echo "[phi4-base-submit] borrow=${BORROW_WS:-<none>}${BORROW_WS:+ ngpu=$BORROW_NGPU}"
echo "[phi4-base-submit] config_sha256=$CONFIG_SHA256"
[[ -n "$OVERLAY_SHA256" ]] && echo "[phi4-base-submit] overlay_sha256=$OVERLAY_SHA256"
echo "[phi4-base-submit] tasks_sha256=$TASKS_SHA256"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] validation passed; no Kubernetes resources were changed."
    exit 0
fi

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/phi4-14b-base-cluster-eval.XXXXXX")"
UPLOAD_DIR="$STAGING_PARENT/mini-web-agent"
SUBMIT_LOG="$STAGING_PARENT/submit.log"
cleanup_staging() {
    if [[ -n "${STAGING_PARENT:-}" &&
          -d "$STAGING_PARENT" &&
          "$(basename "$STAGING_PARENT")" == phi4-14b-base-cluster-eval.* ]]; then
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
print(f"[phi4-base-submit] gold_tasks_staged={staged}")
GOLDPY
fi

python - \
    "$UPLOAD_DIR/cluster_eval_assets/provenance.json" \
    "$BASE_CKPT" \
    "$DTYPE" \
    "${CONFIG_SOURCE#"$MINI_WEB_AGENT_DIR/"}" \
    "$CONFIG_SHA256" \
    "$OVERLAY_SHA256" \
    "$TASKS_SHA256" \
    "$TASK_COUNT" \
    "$GPU_COUNT" \
    "$TP" \
    "$DP" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    base_ckpt,
    dtype,
    config_source,
    config_sha256,
    overlay_sha256,
    tasks_sha256,
    task_count,
    gpu_count,
    tensor_parallel_size,
    data_parallel_size,
) = sys.argv[1:]
Path(output_path).write_text(
    json.dumps(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "base",
            "base_checkpoint": base_ckpt,
            "source_training_job": None,
            "checkpoint_step": None,
            "dtype": dtype,
            "config_source": config_source,
            "config_sha256": config_sha256,
            "overlay_sha256": overlay_sha256 or None,
            "tasks_sha256": tasks_sha256,
            "task_count": int(task_count),
            "node_count": 1,
            "gpu_count": int(gpu_count),
            "tensor_parallel_size": int(tensor_parallel_size),
            "data_parallel_size": int(data_parallel_size),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-${USER_ALIAS}-webchain-sampling-creds}"
mwa_apply_credentials_secret "$NAMESPACE" "$CREDENTIALS_SECRET" "$CREDENTIALS_FILE"

EXTRA_ENV="EVAL_RUN_ID=$EVAL_RUN_ID,BASE_CKPT=$BASE_CKPT"
EXTRA_ENV+=",MODEL_NAME=$MODEL_NAME,DTYPE=$DTYPE"
EXTRA_ENV+=",WORKERS=$WORKERS,JUDGE_NUM_PROC=$JUDGE_NUM_PROC,TP=$TP,DP=$DP"
EXTRA_ENV+=",GPU_COUNT=$GPU_COUNT"
EXTRA_ENV+=",MAX_MODEL_LEN=$MAX_MODEL_LEN,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
EXTRA_ENV+=",CONFIG_SHA256=$CONFIG_SHA256"
[[ -n "$OVERLAY_SHA256" ]] && EXTRA_ENV+=",OVERLAY_SHA256=$OVERLAY_SHA256"
EXTRA_ENV+=",TASKS_SHA256=$TASKS_SHA256"
[[ -n "${MAX_OUTPUT_TOKENS:-}" ]] && EXTRA_ENV+=",MAX_OUTPUT_TOKENS=$MAX_OUTPUT_TOKENS"
[[ -n "${CONTEXT_MARGIN:-}" ]] && EXTRA_ENV+=",CONTEXT_MARGIN=$CONTEXT_MARGIN"
[[ -n "${MAX_CONTEXT_TOKENS:-}" ]] && EXTRA_ENV+=",MAX_CONTEXT_TOKENS=$MAX_CONTEXT_TOKENS"
[[ -n "$STEP_LIMIT" ]] && EXTRA_CFG="agent.step_limit=$STEP_LIMIT${EXTRA_CFG:+ $EXTRA_CFG}"
[[ -n "$EXTRA_CFG" ]] && EXTRA_ENV+=",EXTRA_CFG=$EXTRA_CFG"
if [[ -n "${HF_TOKEN:-}" ]]; then
    EXTRA_ENV+=",HF_TOKEN=$HF_TOKEN,HF_HUB_TOKEN=$HF_TOKEN"
fi

FOLLOW_ARGS=()
[[ "$FOLLOW_LOGS" == "1" ]] && FOLLOW_ARGS=(--follow-logs)

# Resources: the 512Gi/64cpu the sibling launchers ask for is NOT enough for this
# shape and OOMKilled run 6e5ec (exit 137) nine minutes into generation, with
# vLLM healthy and 20/300 tasks already scored. An 8-GPU node here has 2896Gi and
# 110 CPU allocatable, and because this job holds all 8 GPUs nothing else can
# schedule beside it, so taking the bulk of the node's RAM costs nothing. 2600Gi
# / 104 CPU is what the comparable 8-GPU agentic eval on the same node class
# (andrewzhao-p0-cua-eval-tb2-unified-tmax9b) requests and completes on.
set +e
bash "$SUBMIT" \
    --upload "$UPLOAD_DIR" \
    --image "${IMAGE:-aifrontiers.azurecr.io/nvidia-26.06-pytorch-2.13.0-torchao-0.17.0-te-2.17-deepspeed-0.19.2-fa2-77aacb6-fa4-4.0.0b19-vllm-0.25.1:20260717}" \
    --acr \
    --node 1 \
    --gpu-per-node "$GPU_COUNT" \
    --cpu "${EVAL_CPU:-104}" \
    --memory "${EVAL_MEMORY:-2600Gi}" \
    --shm "${EVAL_SHM:-128Gi}" \
    --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" \
    --extra-env-vars "$EXTRA_ENV" \
    "${FOLLOW_ARGS[@]}" \
    --cmd 'exec bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/scripts/cluster/om2w/phi4_14b_base/run.sh' 2>&1 | tee "$SUBMIT_LOG"
SUBMIT_RC=${PIPESTATUS[0]}
set -e
[[ "$SUBMIT_RC" == "0" ]] || exit "$SUBMIT_RC"

# borrow-ws / borrow-ngpu have no submit_job.sh flag, so they are applied to the
# live Job; the collector re-reads .metadata.labels on every scrape.
if [[ -n "$BORROW_WS" ]]; then
    JOB_FULLNAME="$(awk -F': ' '/^Volcano Job name: /{print $2}' "$SUBMIT_LOG" | tail -n 1)"
    [[ -n "$JOB_FULLNAME" ]] || {
        echo "[error] could not read the Volcano job name from the submission log; apply borrow labels by hand" >&2
        exit 1
    }
    kubectl -n "$NAMESPACE" label job.batch.volcano.sh "$JOB_FULLNAME" \
        "borrow-ws=$BORROW_WS" "borrow-ngpu=$BORROW_NGPU" --overwrite
    echo "[phi4-base-submit] labeled $JOB_FULLNAME borrow-ws=$BORROW_WS borrow-ngpu=$BORROW_NGPU"
fi
