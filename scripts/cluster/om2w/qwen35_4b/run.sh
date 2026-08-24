#!/usr/bin/env bash
# Convert the Qwen3.5-4B (webwright-student) PhiTrain RL actor checkpoint to
# Hugging Face format, serve it with vLLM, and run the SPB persistent-browser
# OM2W benchmark inside one Bonete GPU pod.
#
# Same harness, prompts and judge as scripts/cluster/om2w/qwen38_27b/run.sh; the
# submission wrapper stages the same base config. Only the policy differs: an RL
# Ray-actor DCP checkpoint (<step>/actor/*.distcp) instead of a stock HF model.
#
# The actor holds a text-only Qwen3_5ForCausalLM state dict (model.embed_tokens /
# model.layers / model.norm / lm_head, no visual tower), so the converted directory
# takes its config and tokenizer from the base model and is served with
# --language-model-only -- the same mode PhiTrain's rollout engine used for these
# weights during training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../lib/cluster_runtime.sh
source "$SCRIPT_DIR/../../../lib/cluster_runtime.sh"

: "${DATA_ROOT:?DATA_ROOT is not set}"
: "${JOB_NAME:?JOB_NAME is not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID is not set}"
: "${SOURCE_ROOT:?SOURCE_ROOT is not set}"
: "${CHECKPOINT_STEP:?CHECKPOINT_STEP is not set}"

UPLOAD_ROOT="${UPLOAD_ROOT:-$DATA_ROOT/runs/$JOB_NAME}"
UPLOAD_REPO="${UPLOAD_REPO:-$UPLOAD_ROOT/mini-web-agent}"
PHITRAIN_ROOT="${PHITRAIN_ROOT:-$UPLOAD_ROOT/phitrain}"
LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-qwen35-4b-eval}"
ASSET_SUBDIR="${ASSET_SUBDIR:-cluster_eval_assets}"
BASE_MODEL="${BASE_MODEL:-/mnt/pvc/experiments/luyadong/models/webwright-student}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
TP="${TP:-8}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PORT="${PORT:-8000}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-3600}"
# Space-separated dotted `-c` overrides, appended last so they win. e.g.
#   EXTRA_CFG="agent.step_limit=50 model.max_output_tokens=32768"
EXTRA_CFG="${EXTRA_CFG:-}"

ACTOR_DIR="$SOURCE_ROOT/$CHECKPOINT_STEP/actor"
# Named per step so an earlier checkpoint can never reuse or clobber another's
# conversion, and so a rerun reuses the conversion instead of redoing it.
HF_CKPT="${HF_CKPT:-$SOURCE_ROOT/step_${CHECKPOINT_STEP}_hf}"

[[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]] || {
    echo "[qwen35-4b-eval][error] CHECKPOINT_STEP must be a non-negative integer: $CHECKPOINT_STEP" >&2
    exit 2
}
for numeric_name in TP WORKERS JUDGE_NUM_PROC MAX_MODEL_LEN PORT VLLM_WAIT_SECONDS; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen35-4b-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || {
    echo "[qwen35-4b-eval][error] invalid GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2
    exit 2
}
for required_dir in "$UPLOAD_REPO" "$PHITRAIN_ROOT" "$SOURCE_ROOT" "$ACTOR_DIR" "$BASE_MODEL"; do
    [[ -d "$required_dir" ]] || {
        echo "[qwen35-4b-eval][error] required directory is missing: $required_dir" >&2
        exit 1
    }
done
[[ -f "$ACTOR_DIR/actor_config.json" ]] || {
    echo "[qwen35-4b-eval][error] not a Ray-actor checkpoint, no actor_config.json: $ACTOR_DIR" >&2
    exit 1
}
[[ -f "$CREDS_FILE" ]] || {
    echo "[qwen35-4b-eval][error] credentials file is missing: $CREDS_FILE" >&2
    exit 1
}
case "$LOCAL_REPO" in
    /tmp/mini-web-agent-qwen35-4b-eval*) ;;
    *)
        echo "[qwen35-4b-eval][error] refusing unsafe local staging path: $LOCAL_REPO" >&2
        exit 1
        ;;
esac

mwa_copy_staged_repo "$UPLOAD_REPO" "$LOCAL_REPO"
REPO="$LOCAL_REPO"
ASSET_DIR="$REPO/$ASSET_SUBDIR"
CONFIG_FILE="$ASSET_DIR/merged_config.yaml"
OVERLAY_FILE="$ASSET_DIR/overlay_config.yaml"
TASKS_FILE="$ASSET_DIR/tasks.json"
# Staged only when the submission wrapper was given GOLD_SOURCE. The config's own
# run.gold_trajectory_dir points at a submitting-host path that does not exist in
# this pod, so it is repointed here whenever the assets are present.
GOLD_DIR="$ASSET_DIR/gold"

for required_file in "$CONFIG_FILE" "$TASKS_FILE"; do
    [[ -f "$required_file" ]] || {
        echo "[qwen35-4b-eval][error] staged asset is missing: $required_file" >&2
        exit 1
    }
done

mwa_verify_sha256 qwen35-4b-eval "$CONFIG_FILE" "${CONFIG_SHA256:-}"
# The overlay is optional: the wrapper only stages one when OVERLAY_SOURCE is set.
OVERLAY_SPECS=()
OVERLAY_ARGS=()
if [[ -f "$OVERLAY_FILE" ]]; then
    mwa_verify_sha256 qwen35-4b-eval "$OVERLAY_FILE" "${OVERLAY_SHA256:-}"
    OVERLAY_SPECS=("$OVERLAY_FILE")
    OVERLAY_ARGS=(-c "$OVERLAY_FILE")
fi
GOLD_SPECS=()
GOLD_ARGS=()
if [[ -d "$GOLD_DIR" ]]; then
    GOLD_SPECS=("run.gold_trajectory_dir=$GOLD_DIR")
    GOLD_ARGS=(-c "run.gold_trajectory_dir=$GOLD_DIR")
fi
mwa_verify_sha256 qwen35-4b-eval "$TASKS_FILE" "${TASKS_SHA256:-}"

RUN_ROOT="$DATA_ROOT/evals/$EVAL_RUN_ID"
OUTPUTS_DIR="$RUN_ROOT/outputs"
LOGS_DIR="$RUN_ROOT/logs"
mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR"
cp "$ASSET_DIR/provenance.json" "$RUN_ROOT/provenance.json"

source "$CREDS_FILE"
if [[ -n "${PHYAGI_API_KEY:-}" ]]; then
    export OPENAI_GATEWAY_API_KEY="${OPENAI_GATEWAY_API_KEY:-$PHYAGI_API_KEY}"
fi

export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/agent_runtime${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME"

echo "[qwen35-4b-eval] installing conversion and harness dependencies"
# The cluster image provides most dependencies through Debian or the base Python
# environment. Installing the full dependency graph makes pip try to uninstall
# those record-less packages (notably Debian's Pygments), so install only what
# this non-interactive benchmark imports.
python -m pip install --no-deps -e "$PHITRAIN_ROOT"
python -m pip install --no-deps -e "$REPO"
python -m pip install --no-deps browserbase playwright pyee openai pillow backoff
python -c 'import backoff, browserbase, openai, PIL, playwright, miniswewebagent, phitrain'

EXTRA_CFG_SPECS=()
if [[ -n "$EXTRA_CFG" ]]; then
    read -r -a EXTRA_CFG_SPECS <<<"$EXTRA_CFG"
fi
EXTRA_CFG_ARGS=()
for spec in ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}; do
    EXTRA_CFG_ARGS+=(-c "$spec")
done

mwa_print_effective_config \
    qwen35-4b-eval "$CONFIG_FILE" \
    ${OVERLAY_SPECS+"${OVERLAY_SPECS[@]}"} \
    ${GOLD_SPECS+"${GOLD_SPECS[@]}"} \
    ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}

hf_checkpoint_ready() {
    [[ -f "$HF_CKPT/.phitrain_conversion_complete.json" ]] &&
        [[ -f "$HF_CKPT/config.json" ]] &&
        [[ -f "$HF_CKPT/tokenizer_config.json" ]] &&
        [[ -f "$HF_CKPT/chat_template.jinja" ]] &&
        { [[ -f "$HF_CKPT/model.safetensors" ]] ||
            [[ -f "$HF_CKPT/model.safetensors.index.json" ]]; } &&
        grep -q '"weight_prefix": "language_model."' \
            "$HF_CKPT/.phitrain_conversion_complete.json"
}

if ! hf_checkpoint_ready; then
    HF_TMP="$SOURCE_ROOT/.hf_convert.${JOB_NAME}.partial"
    case "$HF_TMP" in
        "$SOURCE_ROOT"/.hf_convert.*.partial) rm -rf -- "$HF_TMP" ;;
        *)
            echo "[qwen35-4b-eval][error] refusing unsafe conversion temp path: $HF_TMP" >&2
            exit 1
            ;;
    esac
    mkdir -p "$HF_TMP"

    echo "[qwen35-4b-eval] converting Ray-actor checkpoint $ACTOR_DIR"
    # convert_ray_actor_checkpoint is called directly rather than through the
    # phitrain CLI: get_checkpoint_type() misroutes the actor/ directory to the
    # FSDP converter because it contains .distcp files.
    python - "$ACTOR_DIR" "$HF_TMP" <<'PY'
import sys
from pathlib import Path

from phitrain.utils.checkpoint.convert import convert_ray_actor_checkpoint

actor_dir, output_dir = (Path(value) for value in sys.argv[1:])
convert_ray_actor_checkpoint(actor_dir, output_dir)
print(f"[qwen35-4b-eval] converted {actor_dir} -> {output_dir}", flush=True)
PY

    # vLLM still builds Qwen3_5ForConditionalGeneration under
    # --language-model-only -- the flag drops the vision tower but keeps the text
    # backbone at language_model.*, while the actor state dict is flat model.*.
    # PhiTrain's rollout applies the same prefix when it streams actor weights
    # into vLLM.
    # Reference: phitrain/phitrain/rl/rollout/utils.py::get_multimodal_prefix
    echo "[qwen35-4b-eval] re-prefixing weights for the vLLM multimodal wrapper"
    python - "$HF_TMP" <<'PY'
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

from safetensors.torch import load_file, save_file

output_dir = Path(sys.argv[1])
tied = json.loads((output_dir / "config.json").read_text(encoding="utf-8")).get("tie_word_embeddings")
index_path = output_dir / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else None
shards = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]

weight_map = {}
total_size = 0
for shard in shards:
    shard_path = output_dir / shard
    renamed = OrderedDict()
    for name, tensor in load_file(shard_path).items():
        # Tied to embed_tokens, and vLLM exposes no lm_head parameter for it.
        if tied and name == "lm_head.weight":
            continue
        target = f"language_model.{name}"
        renamed[target] = tensor
        weight_map[target] = shard
        total_size += tensor.numel() * tensor.element_size()
    staged_path = shard_path.with_name(shard_path.name + ".renamed")
    save_file(renamed, staged_path, metadata={"format": "pt"})
    os.replace(staged_path, shard_path)
    print(f"[qwen35-4b-eval] re-prefixed {len(renamed)} tensors in {shard}", flush=True)

if index is not None:
    index["weight_map"] = weight_map
    index["metadata"] = {**index.get("metadata", {}), "total_size": total_size}
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
print(f"[qwen35-4b-eval] re-prefixed {len(weight_map)} tensors total", flush=True)
PY

    # convert_ray_actor_checkpoint derives config.json from the training-time
    # actor config, which is the bare text sub-config: no `architectures`, and
    # use_cache disabled for training. The base model directory already holds
    # the config the policy was initialised from and served under during RL, so
    # take it (and the tokenizer, chat template and preprocessor) from there.
    echo "[qwen35-4b-eval] staging config and tokenizer from $BASE_MODEL"
    python - "$BASE_MODEL" "$HF_TMP" <<'PY'
import shutil
import sys
from pathlib import Path

base_model, output_dir = (Path(value) for value in sys.argv[1:])
skipped = {"README.md", "LICENSE", ".gitattributes"}
copied = []
for source in sorted(base_model.iterdir()):
    if not source.is_file() or source.name in skipped:
        continue
    if source.name.startswith(".") or ".safetensors" in source.name:
        continue
    shutil.copy2(source, output_dir / source.name)
    copied.append(source.name)
print(f"[qwen35-4b-eval] staged {len(copied)} files: {', '.join(copied)}", flush=True)
PY

    for required_name in config.json tokenizer_config.json chat_template.jinja; do
        [[ -f "$HF_TMP/$required_name" ]] || {
            echo "[qwen35-4b-eval][error] converted checkpoint has no $required_name" >&2
            exit 1
        }
    done
    if [[ ! -f "$HF_TMP/model.safetensors" ]] &&
        [[ ! -f "$HF_TMP/model.safetensors.index.json" ]]; then
        echo "[qwen35-4b-eval][error] converted checkpoint has no safetensors weights" >&2
        exit 1
    fi

    python - "$HF_TMP/.phitrain_conversion_complete.json" "$ACTOR_DIR" "$BASE_MODEL" "$JOB_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker, actor_dir, base_model, job_name = sys.argv[1:]
Path(marker).write_text(
    json.dumps(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "actor_dir": actor_dir,
            "base_model": base_model,
            "conversion_job": job_name,
            "weight_prefix": "language_model.",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

    if [[ -e "$HF_CKPT" ]]; then
        HF_BACKUP="${HF_CKPT}.incomplete.$(date -u +%Y%m%dT%H%M%SZ).$$"
        echo "[qwen35-4b-eval] preserving incomplete conversion at $HF_BACKUP"
        mv "$HF_CKPT" "$HF_BACKUP"
    fi
    mv "$HF_TMP" "$HF_CKPT"
fi

hf_checkpoint_ready || {
    echo "[qwen35-4b-eval][error] Hugging Face checkpoint validation failed: $HF_CKPT" >&2
    exit 1
}
cp "$HF_CKPT/.phitrain_conversion_complete.json" "$RUN_ROOT/checkpoint_selection.json"

echo "[qwen35-4b-eval] model=$HF_CKPT tp=$TP max_model_len=$MAX_MODEL_LEN"
echo "[qwen35-4b-eval] source_root=$SOURCE_ROOT step=$CHECKPOINT_STEP"
echo "[qwen35-4b-eval] run_id=$EVAL_RUN_ID tasks=$TASKS_FILE workers=$WORKERS"
echo "[qwen35-4b-eval] overlay=${OVERLAY_SPECS[0]:-<none>}"
echo "[qwen35-4b-eval] gold_trajectories=$([[ -d "$GOLD_DIR" ]] && find "$GOLD_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l || echo 0)"
nvidia-smi -L

VLLM_LOG="$LOGS_DIR/vllm.log"
# --language-model-only resolves the multimodal config down to its text config,
# which is the architecture the RL actor's weights actually describe. No
# --reasoning-parser on purpose: it moves the think block into
# `reasoning_content` and strips it from `content`, and the sft_state history
# replay stores `content` as extra.raw_response -- every past thought would be
# lost from the replayed context.
vllm serve "$HF_CKPT" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --language-model-only \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --chat-template "$HF_CKPT/chat_template.jinja" \
    --enable-prefix-caching \
    --trust-remote-code >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    mwa_stop_process "${VLLM_PID:-}"
}
trap cleanup EXIT

if ! mwa_wait_for_vllm \
    qwen35-4b-eval "$VLLM_PID" "$PORT" "$VLLM_WAIT_SECONDS"; then
    tail -200 "$VLLM_LOG" >&2 || true
    exit 1
fi

ENDPOINT="http://127.0.0.1:$PORT/v1"
echo "[qwen35-4b-eval] starting OM2W generation and judge"
cd "$REPO"
set +e
python -m miniswewebagent.run.benchmarks.om2w \
    -c "$CONFIG_FILE" \
    ${OVERLAY_ARGS+"${OVERLAY_ARGS[@]}"} \
    -c "model.endpoint=$ENDPOINT" \
    -c "model.model_name=$MODEL_NAME" \
    -c "environment.credentials_file=$CREDS_FILE" \
    -c "environment.env.PYTHONPATH=$REPO/agent_runtime" \
    -c "run.logs_root=$LOGS_DIR" \
    ${GOLD_ARGS+"${GOLD_ARGS[@]}"} \
    ${EXTRA_CFG_ARGS+"${EXTRA_CFG_ARGS[@]}"} \
    --tasks-file "$TASKS_FILE" \
    --task-level all \
    --workers "$WORKERS" \
    --judge-num-proc "$JUDGE_NUM_PROC" \
    --output-dir "$OUTPUTS_DIR" 2>&1 | tee "$LOGS_DIR/run.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

echo "[qwen35-4b-eval] finished rc=$EVAL_RC run_root=$RUN_ROOT"
exit "$EVAL_RC"
