#!/usr/bin/env bash
# Convert a completed PhiTrain Qwen3.5-9B DCP checkpoint to Hugging Face
# format, serve it with vLLM, and run the frozen 300-task last-observation eval.

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT is not set}"
: "${JOB_NAME:?JOB_NAME is not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID is not set}"
: "${SOURCE_ROOT:?SOURCE_ROOT is not set}"

UPLOAD_ROOT="${UPLOAD_ROOT:-$DATA_ROOT/runs/$JOB_NAME}"
UPLOAD_REPO="${UPLOAD_REPO:-$UPLOAD_ROOT/mini-web-agent}"
PHITRAIN_ROOT="${PHITRAIN_ROOT:-$UPLOAD_ROOT/phitrain}"
LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-qwen35-eval}"
ASSET_SUBDIR="${ASSET_SUBDIR:-cluster_eval_assets}"
# Empty selects the highest-numbered checkpoint (the historical behaviour).
# Set to a step number to evaluate an earlier checkpoint. The converted HF
# directory is named per step, so an earlier step can never reuse (or clobber)
# the final checkpoint's conversion in last_hf.
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
if [[ -n "$CHECKPOINT_STEP" ]]; then
    [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]] || {
        echo "[qwen35-eval][error] CHECKPOINT_STEP must be a non-negative integer: $CHECKPOINT_STEP" >&2
        exit 2
    }
    HF_CKPT="${HF_CKPT:-$SOURCE_ROOT/step_${CHECKPOINT_STEP}_hf}"
else
    HF_CKPT="${HF_CKPT:-$SOURCE_ROOT/last_hf}"
fi
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
TP="${TP:-8}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PORT="${PORT:-8000}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-3600}"
# Empty keeps the frozen config's agent.step_limit; set to override it.
STEP_LIMIT="${STEP_LIMIT:-}"
STEP_LIMIT_ARGS=()
if [[ -n "$STEP_LIMIT" ]]; then
    [[ "$STEP_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen35-eval][error] STEP_LIMIT must be a positive integer: $STEP_LIMIT" >&2
        exit 2
    }
    STEP_LIMIT_ARGS=(-c "agent.step_limit=$STEP_LIMIT")
fi
# The phitrain conversion writes max_position_embeddings from the training
# sequence length (32768) into text_config, while base Qwen3.5-9B is 262144.
# The two text_configs are otherwise identical (rope_scaling=None, same
# head_dim/rope geometry, no sliding_window), so the checkpoint's value is
# metadata rather than an architectural ceiling. --hf-overrides cannot fix it:
# a nested {"text_config": {...}} replaces the sub-config with a plain dict and
# crashes, and a flat key only sets the unused top-level attribute. So use
# vLLM's documented escape hatch instead. Safe here precisely because the base
# RoPE covers 262144; positions past the 32k trained length are still
# extrapolation, so set this deliberately.
ALLOW_LONG_MAX_MODEL_LEN="${ALLOW_LONG_MAX_MODEL_LEN:-0}"
if [[ "$ALLOW_LONG_MAX_MODEL_LEN" == "1" ]]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    echo "[qwen35-eval] VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (serving beyond the checkpoint's max_position_embeddings)"
fi
# Hard ceiling on prompt tokens per request. The agent evicts its oldest
# assistant turns to stay under it, so a long episode degrades gracefully
# instead of vLLM rejecting an over-length request and failing the task. Derived
# from the served context window so the two can never drift apart; set to 0 to
# disable the budget entirely.
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
CONTEXT_MARGIN="${CONTEXT_MARGIN:-672}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-$((MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - CONTEXT_MARGIN))}"
[[ "$MAX_CONTEXT_TOKENS" =~ ^[0-9]+$ ]] || {
    echo "[qwen35-eval][error] MAX_CONTEXT_TOKENS must be a non-negative integer: $MAX_CONTEXT_TOKENS" >&2
    exit 2
}
if (( MAX_CONTEXT_TOKENS > 0 && MAX_CONTEXT_TOKENS + MAX_OUTPUT_TOKENS > MAX_MODEL_LEN )); then
    echo "[qwen35-eval][error] MAX_CONTEXT_TOKENS + MAX_OUTPUT_TOKENS exceeds MAX_MODEL_LEN" >&2
    exit 2
fi

for numeric_name in TP WORKERS JUDGE_NUM_PROC MAX_MODEL_LEN PORT VLLM_WAIT_SECONDS; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[qwen35-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || {
    echo "[qwen35-eval][error] invalid GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2
    exit 2
}
for required_dir in "$UPLOAD_REPO" "$PHITRAIN_ROOT" "$SOURCE_ROOT"; do
    [[ -d "$required_dir" ]] || {
        echo "[qwen35-eval][error] required directory is missing: $required_dir" >&2
        exit 1
    }
done
[[ -f "$CREDS_FILE" ]] || {
    echo "[qwen35-eval][error] credentials file is missing: $CREDS_FILE" >&2
    exit 1
}
case "$LOCAL_REPO" in
    /tmp/mini-web-agent-qwen35-eval*) ;;
    *)
        echo "[qwen35-eval][error] refusing unsafe local staging path: $LOCAL_REPO" >&2
        exit 1
        ;;
esac

rm -rf "$LOCAL_REPO"
mkdir -p "$LOCAL_REPO"
cp -R --no-preserve=mode,ownership,timestamps "$UPLOAD_REPO/." "$LOCAL_REPO/"
REPO="$LOCAL_REPO"
ASSET_DIR="$REPO/$ASSET_SUBDIR"
CONFIG_FILE="$ASSET_DIR/merged_config.yaml"
TASKS_FILE="$ASSET_DIR/tasks.json"
CHAT_TEMPLATE="$ASSET_DIR/qwen3_5_no_auto_think.jinja"

for required_file in "$CONFIG_FILE" "$TASKS_FILE" "$CHAT_TEMPLATE"; do
    [[ -f "$required_file" ]] || {
        echo "[qwen35-eval][error] staged asset is missing: $required_file" >&2
        exit 1
    }
done

verify_sha256() {
    local file_path="$1"
    local expected="$2"
    local actual
    [[ -z "$expected" ]] && return 0
    actual="$(sha256sum "$file_path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "[qwen35-eval][error] checksum mismatch for $file_path" >&2
        echo "[qwen35-eval][error] expected=$expected actual=$actual" >&2
        exit 1
    }
}
verify_sha256 "$CONFIG_FILE" "${CONFIG_SHA256:-}"
verify_sha256 "$TASKS_FILE" "${TASKS_SHA256:-}"
verify_sha256 "$CHAT_TEMPLATE" "${CHAT_TEMPLATE_SHA256:-}"

RUN_ROOT="$DATA_ROOT/evals/$EVAL_RUN_ID"
OUTPUTS_DIR="$RUN_ROOT/outputs"
LOGS_DIR="$RUN_ROOT/logs"
mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR"
cp "$ASSET_DIR/provenance.json" "$RUN_ROOT/provenance.json"

source "$CREDS_FILE"
if [[ -n "${PHYAGI_API_KEY:-}" ]]; then
    export OPENAI_GATEWAY_API_KEY="${OPENAI_GATEWAY_API_KEY:-$PHYAGI_API_KEY}"
fi
if [[ -n "${HUGGINGFACE_API_KEY:-}" ]]; then
    export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_API_KEY}"
    export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HUGGINGFACE_API_KEY}"
fi

export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/agent_runtime${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME"

echo "[qwen35-eval] installing conversion and harness packages"
python -m pip install --no-deps -e "$PHITRAIN_ROOT"
python -m pip install --no-deps -e "$REPO"
python -m pip install --no-deps browserbase playwright pyee openai pillow backoff
python -c 'import backoff, browserbase, openai, PIL, playwright, miniswewebagent, phitrain'

hf_checkpoint_ready() {
    [[ -f "$HF_CKPT/.phitrain_conversion_complete.json" ]] &&
        [[ -f "$HF_CKPT/config.json" ]] &&
        [[ -f "$HF_CKPT/tokenizer_config.json" ]] &&
        [[ -f "$HF_CKPT/preprocessor_config.json" ]] &&
        { [[ -f "$HF_CKPT/model.safetensors" ]] ||
            [[ -f "$HF_CKPT/model.safetensors.index.json" ]]; }
}

if ! hf_checkpoint_ready; then
    HF_TMP="$SOURCE_ROOT/.hf_convert.${JOB_NAME}.partial"
    case "$HF_TMP" in
        "$SOURCE_ROOT"/.hf_convert.*.partial) rm -rf -- "$HF_TMP" ;;
        *)
            echo "[qwen35-eval][error] refusing unsafe conversion temp path: $HF_TMP" >&2
            exit 1
            ;;
    esac
    mkdir -p "$HF_TMP"

    echo "[qwen35-eval] converting DCP checkpoint under $SOURCE_ROOT (step=${CHECKPOINT_STEP:-latest})"
    python - "$SOURCE_ROOT" "$HF_TMP" "${CHECKPOINT_STEP:-}" <<'PY'
import json
import sys
from pathlib import Path

from phitrain.utils.checkpoint.convert import bundle_tokenizer, convert_fsdp_checkpoint
from phitrain.utils.checkpoint.info import find_final_checkpoint

source_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
requested_step = (sys.argv[3] if len(sys.argv) > 3 else "").strip()
found = find_final_checkpoint(source_root)
if found is None and not requested_step:
    raise SystemExit(f"[qwen35-eval][error] no final checkpoint under {source_root}")

# This run's Lightning callback updates `last/` only at save_steps. Prefer the
# highest numbered state so metrics written after that save cannot make `last`
# appear newer than the tensors it actually contains.
numbered = sorted(
    (path for path in source_root.iterdir() if path.is_dir() and path.name.isdigit()),
    key=lambda path: int(path.name),
)
if requested_step:
    # Explicit step wins, so an earlier checkpoint can be evaluated without
    # depending on whichever one happens to be newest.
    matches = [path for path in numbered if int(path.name) == int(requested_step)]
    if not matches:
        available = ", ".join(path.name for path in numbered) or "<none>"
        raise SystemExit(
            f"[qwen35-eval][error] checkpoint step {requested_step} not found under "
            f"{source_root}; available: {available}"
        )
    source_checkpoint = matches[0]
    checkpoint_basename = source_checkpoint.name
    checkpoint_step = int(checkpoint_basename)
elif numbered:
    source_checkpoint = numbered[-1]
    checkpoint_basename = source_checkpoint.name
    checkpoint_step = int(checkpoint_basename)
else:
    source_checkpoint = Path(found["path"])
    checkpoint_basename = str(found["basename"])
    checkpoint_step = int(found["step"])
print(
    f"[qwen35-eval] selected checkpoint={source_checkpoint} "
    f"step={checkpoint_step}",
    flush=True,
)
convert_fsdp_checkpoint(source_checkpoint, output_dir)
bundle_tokenizer(output_dir, source_root=source_root)
(output_dir / "checkpoint_selection.json").write_text(
    json.dumps(
        {
            "source_root": str(source_root),
            "source_checkpoint": str(source_checkpoint),
            "checkpoint_basename": checkpoint_basename,
            "checkpoint_step": checkpoint_step,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

    python - "$BASE_MODEL" "$HF_TMP" <<'PY'
import shutil
import sys

from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

base_model, output_dir = sys.argv[1:]
processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
processor.save_pretrained(output_dir)
for file_name in ("preprocessor_config.json", "video_preprocessor_config.json"):
    source = hf_hub_download(repo_id=base_model, filename=file_name)
    shutil.copy2(source, output_dir)
PY

    [[ -f "$HF_TMP/config.json" ]] || {
        echo "[qwen35-eval][error] converted checkpoint has no config.json" >&2
        exit 1
    }
    [[ -f "$HF_TMP/tokenizer_config.json" ]] || {
        echo "[qwen35-eval][error] converted checkpoint has no tokenizer_config.json" >&2
        exit 1
    }
    [[ -f "$HF_TMP/preprocessor_config.json" ]] || {
        echo "[qwen35-eval][error] converted checkpoint has no preprocessor_config.json" >&2
        exit 1
    }
    if [[ ! -f "$HF_TMP/model.safetensors" ]] &&
        [[ ! -f "$HF_TMP/model.safetensors.index.json" ]]; then
        echo "[qwen35-eval][error] converted checkpoint has no safetensors weights" >&2
        exit 1
    fi

    python - "$HF_TMP/.phitrain_conversion_complete.json" "$SOURCE_ROOT" "$JOB_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker, source_root, job_name = sys.argv[1:]
Path(marker).write_text(
    json.dumps(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": source_root,
            "conversion_job": job_name,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

    if [[ -e "$HF_CKPT" ]]; then
        HF_BACKUP="${HF_CKPT}.incomplete.$(date -u +%Y%m%dT%H%M%SZ).$$"
        echo "[qwen35-eval] preserving incomplete conversion at $HF_BACKUP"
        mv "$HF_CKPT" "$HF_BACKUP"
    fi
    mv "$HF_TMP" "$HF_CKPT"
fi

hf_checkpoint_ready || {
    echo "[qwen35-eval][error] Hugging Face checkpoint validation failed: $HF_CKPT" >&2
    exit 1
}

[[ -f "$HF_CKPT/checkpoint_selection.json" ]] && {
    cp "$HF_CKPT/checkpoint_selection.json" "$RUN_ROOT/checkpoint_selection.json"
}
echo "[qwen35-eval] model=$HF_CKPT tp=$TP max_model_len=$MAX_MODEL_LEN"
echo "[qwen35-eval] context_budget=$MAX_CONTEXT_TOKENS max_output_tokens=$MAX_OUTPUT_TOKENS"
echo "[qwen35-eval] run_id=$EVAL_RUN_ID tasks=$TASKS_FILE workers=$WORKERS"
nvidia-smi -L

VLLM_LOG="$LOGS_DIR/vllm.log"
vllm serve "$HF_CKPT" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --chat-template "$CHAT_TEMPLATE" \
    --enable-prefix-caching \
    --trust-remote-code >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if ! python - "$VLLM_PID" "$PORT" "$VLLM_WAIT_SECONDS" <<'PY'
import sys
import time
import urllib.request
from pathlib import Path

pid, port, timeout = map(int, sys.argv[1:])
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print(f"[qwen35-eval] vLLM ready: {url}", flush=True)
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        if not Path(f"/proc/{pid}/stat").exists():
            raise SystemExit("[qwen35-eval][error] vLLM exited before readiness")
        time.sleep(5)
raise SystemExit(f"[qwen35-eval][error] vLLM readiness timed out: {url}")
PY
then
    tail -200 "$VLLM_LOG" >&2 || true
    exit 1
fi

ENDPOINT="http://127.0.0.1:$PORT/v1"
echo "[qwen35-eval] starting OM2W generation and judge"
cd "$REPO"
set +e
python -m miniswewebagent.run.benchmarks.om2w \
    -c "$CONFIG_FILE" \
    -c "model.endpoint=$ENDPOINT" \
    -c "model.model_name=$MODEL_NAME" \
    -c "environment.credentials_file=$CREDS_FILE" \
    -c "environment.env.PYTHONPATH=$REPO/agent_runtime" \
    -c "run.logs_root=$LOGS_DIR" \
    -c "model.max_output_tokens=$MAX_OUTPUT_TOKENS" \
    -c "agent.max_context_tokens=$MAX_CONTEXT_TOKENS" \
    "${STEP_LIMIT_ARGS[@]}" \
    --tasks-file "$TASKS_FILE" \
    --task-level all \
    --workers "$WORKERS" \
    --judge-num-proc "$JUDGE_NUM_PROC" \
    --output-dir "$OUTPUTS_DIR" 2>&1 | tee "$LOGS_DIR/run.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

echo "[qwen35-eval] finished rc=$EVAL_RC run_root=$RUN_ROOT"
exit "$EVAL_RC"
