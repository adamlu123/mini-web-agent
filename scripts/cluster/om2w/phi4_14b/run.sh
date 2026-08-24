#!/usr/bin/env bash
# Serve the Phi-4-14B webwright SFT checkpoint with vLLM and run the SPB
# persistent-browser OM2W benchmark inside one Bonete GPU pod.
#
# Same harness, prompts and judge as scripts/cluster/om2w/qwen35_4b/run.sh; the
# submission wrapper stages the same base config (eval/om2w_spb_vllm_lastobs.yaml).
# Only the policy differs.
#
# The checkpoint is the ALREADY-CONVERTED Hugging Face export the training job
# left at <SOURCE_ROOT>/last_hf: LlamaForCausalLM, bf16, 7 safetensors shards.
# That export carries ONLY config.json + weights + index -- no tokenizer, no chat
# template, no generation config. So this script assembles a serving directory
# that symlinks the weights (29 GB; never copied) and takes the tokenizer, chat
# template and generation config from the model the run was trained from.
#
# Two Phi-4 specifics that differ from the Qwen launchers:
#
#   1. TP MUST DIVIDE num_key_value_heads=10. vLLM shards KV heads across ranks
#      and needs total_num_kv_heads % tp == 0 (or tp % total_num_kv_heads == 0),
#      so TP=8 is IMPOSSIBLE here -- only 1, 2 and 5 work on an 8-GPU node.
#      The node is filled with data parallelism instead: TP=2 x DP=4 = 8 GPUs,
#      four independent replicas behind one API port.
#   2. max_position_embeddings is 32768, not 131072. Long agentic episodes will
#      run into that, so the agent-level token budget (agent.max_context_tokens)
#      is derived from the served window and passed through -- the agent evicts
#      its oldest assistant turns instead of vLLM 400ing an over-length request
#      and failing the task outright.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../lib/cluster_runtime.sh
source "$SCRIPT_DIR/../../../lib/cluster_runtime.sh"

: "${DATA_ROOT:?DATA_ROOT is not set}"
: "${JOB_NAME:?JOB_NAME is not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID is not set}"
: "${HF_CKPT:?HF_CKPT is not set}"
: "${BASE_MODEL:?BASE_MODEL is not set}"

UPLOAD_ROOT="${UPLOAD_ROOT:-$DATA_ROOT/runs/$JOB_NAME}"
UPLOAD_REPO="${UPLOAD_REPO:-$UPLOAD_ROOT/mini-web-agent}"
LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-phi4-14b-eval}"
ASSET_SUBDIR="${ASSET_SUBDIR:-cluster_eval_assets}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
TP="${TP:-2}"
DP="${DP:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
WORKERS="${WORKERS:-80}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
# config.json max_position_embeddings for this checkpoint.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
PORT="${PORT:-8000}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-3600}"
# Space-separated dotted `-c` overrides, appended last so they win. e.g.
#   EXTRA_CFG="agent.step_limit=50 model.max_output_tokens=8192"
EXTRA_CFG="${EXTRA_CFG:-}"

for numeric_name in TP DP GPU_COUNT WORKERS JUDGE_NUM_PROC MAX_MODEL_LEN PORT VLLM_WAIT_SECONDS; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[phi4-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
        exit 2
    }
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || {
    echo "[phi4-eval][error] invalid GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2
    exit 2
}
if (( TP * DP > GPU_COUNT )); then
    echo "[phi4-eval][error] TP*DP=$((TP * DP)) exceeds the pod's $GPU_COUNT GPUs" >&2
    exit 2
fi
for required_dir in "$UPLOAD_REPO" "$HF_CKPT" "$BASE_MODEL"; do
    [[ -d "$required_dir" ]] || {
        echo "[phi4-eval][error] required directory is missing: $required_dir" >&2
        exit 1
    }
done
[[ -f "$CREDS_FILE" ]] || {
    echo "[phi4-eval][error] credentials file is missing: $CREDS_FILE" >&2
    exit 1
}
case "$LOCAL_REPO" in
    /tmp/mini-web-agent-phi4-14b-eval*) ;;
    *)
        echo "[phi4-eval][error] refusing unsafe local staging path: $LOCAL_REPO" >&2
        exit 1
        ;;
esac

# Hard ceiling on prompt tokens per request, derived from the served context
# window so the two can never drift apart. Set MAX_CONTEXT_TOKENS=0 to disable
# the budget entirely and let over-window requests fail the task.
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
CONTEXT_MARGIN="${CONTEXT_MARGIN:-672}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-$((MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - CONTEXT_MARGIN))}"
[[ "$MAX_CONTEXT_TOKENS" =~ ^[0-9]+$ ]] || {
    echo "[phi4-eval][error] MAX_CONTEXT_TOKENS must be a non-negative integer: $MAX_CONTEXT_TOKENS" >&2
    exit 2
}
if (( MAX_CONTEXT_TOKENS > 0 && MAX_CONTEXT_TOKENS + MAX_OUTPUT_TOKENS > MAX_MODEL_LEN )); then
    echo "[phi4-eval][error] MAX_CONTEXT_TOKENS + MAX_OUTPUT_TOKENS exceeds MAX_MODEL_LEN" >&2
    exit 2
fi

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
        echo "[phi4-eval][error] staged asset is missing: $required_file" >&2
        exit 1
    }
done

mwa_verify_sha256 phi4-eval "$CONFIG_FILE" "${CONFIG_SHA256:-}"
# The overlay is optional: the wrapper only stages one when OVERLAY_SOURCE is set.
OVERLAY_SPECS=()
OVERLAY_ARGS=()
if [[ -f "$OVERLAY_FILE" ]]; then
    mwa_verify_sha256 phi4-eval "$OVERLAY_FILE" "${OVERLAY_SHA256:-}"
    OVERLAY_SPECS=("$OVERLAY_FILE")
    OVERLAY_ARGS=(-c "$OVERLAY_FILE")
fi
GOLD_SPECS=()
GOLD_ARGS=()
if [[ -d "$GOLD_DIR" ]]; then
    GOLD_SPECS=("run.gold_trajectory_dir=$GOLD_DIR")
    GOLD_ARGS=(-c "run.gold_trajectory_dir=$GOLD_DIR")
fi
mwa_verify_sha256 phi4-eval "$TASKS_FILE" "${TASKS_SHA256:-}"

RUN_ROOT="$DATA_ROOT/evals/$EVAL_RUN_ID"
OUTPUTS_DIR="$RUN_ROOT/outputs"
LOGS_DIR="$RUN_ROOT/logs"
SERVE_DIR="${SERVE_DIR:-$RUN_ROOT/serve}"
mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR"
cp "$ASSET_DIR/provenance.json" "$RUN_ROOT/provenance.json"

# Sample container memory every MEMORY_SAMPLE_SECONDS into logs/memory.log. Run
# 6e5ec was OOMKilled (exit 137) with no record of what had grown, which left
# nothing to size the limit from; a few bytes a minute makes the next one
# answerable. Killed by the EXIT trap installed further down.
MEMORY_SAMPLE_SECONDS="${MEMORY_SAMPLE_SECONDS:-30}"
MEMORY_LOG="$LOGS_DIR/memory.log"
if (( MEMORY_SAMPLE_SECONDS > 0 )); then
    (
        while true; do
            {
                printf '=== %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                if [[ -r /sys/fs/cgroup/memory.current ]]; then
                    printf 'cgroup.current=%s cgroup.max=%s\n' \
                        "$(cat /sys/fs/cgroup/memory.current 2>/dev/null)" \
                        "$(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
                    grep -E '^(anon|file|slab|shmem|pgfault) ' /sys/fs/cgroup/memory.stat 2>/dev/null
                elif [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
                    printf 'cgroup.usage=%s cgroup.limit=%s\n' \
                        "$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)" \
                        "$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"
                fi
                free -g 2>/dev/null | head -2
                printf 'top RSS (MB):\n'
                ps -eo rss=,comm=,args= --sort=-rss 2>/dev/null |
                    head -12 |
                    awk '{printf "  %8.0f %s %s %s\n", $1/1024, $2, $3, $4}'
                printf 'process count: %s\n' "$(ps -e --no-headers 2>/dev/null | wc -l)"
            } >>"$MEMORY_LOG" 2>&1
            sleep "$MEMORY_SAMPLE_SECONDS"
        done
    ) &
    MEMORY_PID=$!
    echo "[phi4-eval] sampling memory every ${MEMORY_SAMPLE_SECONDS}s -> $MEMORY_LOG"
fi

# Installed here rather than after `vllm serve` so the sampler is reaped even if
# the script dies during setup. Both pids are read lazily, so VLLM_PID being
# unset until later is fine.
cleanup() {
    mwa_stop_process "${VLLM_PID:-}"
    mwa_stop_process "${MEMORY_PID:-}"
}
trap cleanup EXIT

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

echo "[phi4-eval] installing harness dependencies"
# The cluster image provides most dependencies through Debian or the base Python
# environment. Installing the full dependency graph makes pip try to uninstall
# those record-less packages (notably Debian's Pygments), so install only what
# this non-interactive benchmark imports.
python -m pip install --no-deps -e "$REPO"
python -m pip install --no-deps browserbase playwright pyee openai pillow backoff
python -c 'import backoff, browserbase, openai, PIL, playwright, miniswewebagent'

# ---------------------------------------------------------------------------
# Assemble the serving directory.
#
# last_hf holds config.json + model-*.safetensors + model.safetensors.index.json
# and nothing else, so vLLM cannot build a tokenizer from it. The weights are
# SYMLINKED (29 GB stays exactly where it is) and the four small text assets are
# COPIED from the base model the run was fine-tuned from. Rebuilt from scratch on
# every run so a stale or half-written directory can never be served.
# ---------------------------------------------------------------------------
echo "[phi4-eval] assembling serving directory $SERVE_DIR"
case "$SERVE_DIR" in
    "$RUN_ROOT"/*) rm -rf -- "$SERVE_DIR" ;;
    *) echo "[phi4-eval] SERVE_DIR is outside the run root; leaving existing contents in place" ;;
esac
mkdir -p "$SERVE_DIR"
python - "$HF_CKPT" "$BASE_MODEL" "$SERVE_DIR" <<'PY'
import shutil
import sys
from pathlib import Path

checkpoint, base_model, serve_dir = (Path(value) for value in sys.argv[1:])

# From the checkpoint: config.json (the trained architecture, dtype bfloat16 and
# the 32768 position ceiling) plus every weight shard, by symlink.
weights = sorted(checkpoint.glob("*.safetensors"))
if not weights:
    raise SystemExit(f"[phi4-eval][error] no safetensors weights under {checkpoint}")
for name in ("config.json", "model.safetensors.index.json"):
    source = checkpoint / name
    if source.is_file():
        shutil.copy2(source, serve_dir / name)
if not (serve_dir / "config.json").is_file():
    raise SystemExit(f"[phi4-eval][error] {checkpoint} has no config.json")
for source in weights:
    (serve_dir / source.name).symlink_to(source)

# From the base model: the tokenizer and chat template the run was trained with.
# configuration_llama.py / modeling_llama.py are deliberately NOT staged -- they
# are not referenced by an `auto_map` in config.json, so nothing reads them and
# staging them would only invite a trust_remote_code path.
wanted = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
staged = []
for name in wanted:
    source = base_model / name
    if source.is_file():
        shutil.copy2(source, serve_dir / name)
        staged.append(name)
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if not (serve_dir / name).is_file():
        raise SystemExit(
            f"[phi4-eval][error] base model {base_model} has no {name}"
        )
print(
    f"[phi4-eval] serving dir: {len(weights)} weight shard(s) symlinked, "
    f"staged {', '.join(staged)}",
    flush=True,
)
PY

EXTRA_CFG_SPECS=()
if [[ -n "$EXTRA_CFG" ]]; then
    read -r -a EXTRA_CFG_SPECS <<<"$EXTRA_CFG"
fi
BUDGET_SPECS=(
    "model.max_output_tokens=$MAX_OUTPUT_TOKENS"
    "agent.max_context_tokens=$MAX_CONTEXT_TOKENS"
)
EXTRA_CFG_ARGS=()
for spec in ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}; do
    EXTRA_CFG_ARGS+=(-c "$spec")
done

mwa_print_effective_config \
    phi4-eval "$CONFIG_FILE" \
    ${OVERLAY_SPECS+"${OVERLAY_SPECS[@]}"} \
    ${GOLD_SPECS+"${GOLD_SPECS[@]}"} \
    "${BUDGET_SPECS[@]}" \
    ${EXTRA_CFG_SPECS+"${EXTRA_CFG_SPECS[@]}"}

echo "[phi4-eval] checkpoint=$HF_CKPT base_model=$BASE_MODEL"
echo "[phi4-eval] serve_dir=$SERVE_DIR tp=$TP dp=$DP max_model_len=$MAX_MODEL_LEN"
echo "[phi4-eval] context_budget=$MAX_CONTEXT_TOKENS max_output_tokens=$MAX_OUTPUT_TOKENS"
echo "[phi4-eval] run_id=$EVAL_RUN_ID tasks=$TASKS_FILE workers=$WORKERS"
echo "[phi4-eval] overlay=${OVERLAY_SPECS[0]:-<none>}"
echo "[phi4-eval] gold_trajectories=$([[ -d "$GOLD_DIR" ]] && find "$GOLD_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l || echo 0)"
nvidia-smi -L

VLLM_LOG="$LOGS_DIR/vllm.log"
# --data-parallel-size runs DP independent engine replicas behind this one API
# port, each sharded over TP GPUs; it is how the node gets filled at all, since
# num_key_value_heads=10 rules out TP=8. No --trust-remote-code: config.json is
# plain model_type=llama with no auto_map. No --reasoning-parser on purpose: it
# moves the think block into `reasoning_content` and strips it from `content`,
# and the sft_state history replay stores `content` as extra.raw_response --
# every past thought would be lost from the replayed context.
vllm serve "$SERVE_DIR" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --chat-template "$SERVE_DIR/chat_template.jinja" \
    --enable-prefix-caching >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

if ! mwa_wait_for_vllm \
    phi4-eval "$VLLM_PID" "$PORT" "$VLLM_WAIT_SECONDS"; then
    tail -200 "$VLLM_LOG" >&2 || true
    exit 1
fi

ENDPOINT="http://127.0.0.1:$PORT/v1"
echo "[phi4-eval] starting OM2W generation and judge"
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
    -c "model.max_output_tokens=$MAX_OUTPUT_TOKENS" \
    -c "agent.max_context_tokens=$MAX_CONTEXT_TOKENS" \
    ${GOLD_ARGS+"${GOLD_ARGS[@]}"} \
    ${EXTRA_CFG_ARGS+"${EXTRA_CFG_ARGS[@]}"} \
    --tasks-file "$TASKS_FILE" \
    --task-level all \
    --workers "$WORKERS" \
    --judge-num-proc "$JUDGE_NUM_PROC" \
    --output-dir "$OUTPUTS_DIR" 2>&1 | tee "$LOGS_DIR/run.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

echo "[phi4-eval] finished rc=$EVAL_RC run_root=$RUN_ROOT"
exit "$EVAL_RC"
