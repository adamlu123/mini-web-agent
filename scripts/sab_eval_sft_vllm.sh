#!/usr/bin/env bash
# Evaluate a science-SFT (D3-Gym-trained) checkpoint on ScienceAgentBench through
# the miniswewebagent harness: serve the model with vLLM, run the SAB tasks with
# the sft_state agent, collect final_script.py -> pred_programs, then hand off to
# ScienceAgentBench's own evaluator for scoring.
#
# Stage A (this script): inference. Reproduces the vLLM-serving pattern of
#   scripts/mini_harness_eval_sft_vllm.sh but points the batch runner at
#   sab_verified.json + sab_science_sft_vllm.yaml (judge disabled). Produces one
#   final_script.py per task and a pred_programs/ dir.
# Stage B (SAB harness, NOT run here): scoring. Run inside the ScienceAgentBench
#   repo with its own conda env. The exact command is printed at the end. It needs
#   docker (dockerized eval) or the sci-agent-eval conda env (direct eval), which
#   live on a machine with the SAB benchmark installed — usually this same box,
#   not the GPU training cluster.
#
# Examples:
#   SMOKE=1 CKPT=/data/t-yifeili/ckpts/eval_st_1ep bash scripts/sab_eval_sft_vllm.sh
#   CKPT=/mnt/pvc/.../models/qwen35_9b/full/d3gym_32b TP=4 WORKERS=8 \
#     bash scripts/sab_eval_sft_vllm.sh
#   START_VLLM=0 ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
#     MODEL_NAME=d3gym_32b LIMIT=5 bash scripts/sab_eval_sft_vllm.sh
set -euo pipefail

REPO="${REPO:-/data/t-yifeili/mini-web-agent}"
PY="${PY:-/data/t-yifeili/miniconda3/envs/echo-rl/bin/python}"
SAB_REPO="${SAB_REPO:-/data/t-yifeili/ScienceAgentBench}"
SAB_BENCHMARK_DATASETS="${SAB_BENCHMARK_DATASETS:-$SAB_REPO/benchmark/datasets}"

CKPT="${CKPT:-Qwen/Qwen3.5-9B}"
if [[ "$CKPT" == "Qwen/Qwen3.5-9B" ]]; then
  DEFAULT_MODEL_NAME="qwen35_9b_base"
else
  DEFAULT_MODEL_NAME="$(basename "$CKPT")"
fi
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-http://${HOST}:${PORT}/v1/chat/completions}"

TASKS_FILE="${TASKS_FILE:-src/miniswewebagent/run/benchmarks/sab_verified.json}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/sab_science_sft_vllm.yaml}"
TASK_LEVEL="${TASK_LEVEL:-sab}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-4}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/sab_sft/${MODEL_NAME}_${STAMP}}"
PRED_OUT="${PRED_OUT:-$SAB_REPO/pred_programs_${MODEL_NAME}_${STAMP}}"
RUN_LOG="${RUN_LOG:-$SAB_REPO/sab_${MODEL_NAME}_${STAMP}_run.jsonl}"

START_VLLM="${START_VLLM:-1}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$PY")/vllm}"
[[ -x "$VLLM_BIN" ]] || VLLM_BIN="vllm"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$OUTPUT_DIR/vllm.log}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  LIMIT="${LIMIT_SMOKE:-2}"
  WORKERS=1
fi

if [[ ! -d "$SAB_BENCHMARK_DATASETS" ]]; then
  echo "[error] SAB datasets not found: $SAB_BENCHMARK_DATASETS"
  echo "        Set SAB_REPO/SAB_BENCHMARK_DATASETS to your ScienceAgentBench benchmark checkout."
  exit 1
fi

cd "$REPO"
"$PY" -m pip install -e . --no-deps >/dev/null

mkdir -p "$OUTPUT_DIR"

VLLM_PID=""
cleanup() { [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
  echo "[vllm] serving $CKPT as $MODEL_NAME on $HOST:$PORT (log: $VLLM_LOG_FILE)"
  "$VLLM_BIN" serve "$CKPT" \
    ${CHAT_TEMPLATE:+--chat-template "$CHAT_TEMPLATE"} \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    ${VLLM_ARGS:-} >"$VLLM_LOG_FILE" 2>&1 &
  VLLM_PID=$!

  echo "[vllm] waiting for $ENDPOINT"
  "$PY" - <<PY
import sys, time, urllib.request
url = "http://${HOST}:${PORT}/v1/models"
deadline = time.time() + int("${VLLM_WAIT_SECONDS}")
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status < 500:
                print("[vllm] ready"); raise SystemExit(0)
    except SystemExit: raise
    except Exception:
        time.sleep(5)
print(f"[error] vLLM not ready before deadline: {url}", file=sys.stderr); raise SystemExit(1)
PY
fi

# GEN_MODE selects how Stage A generates programs:
#   singleturn (default) : one chat completion per task; extract the program from
#                          <bash>...</bash> OR a ```python``` fence (scripts/
#                          sab_infer_singleturn.py). Matches single-turn D3 SFT data.
#   agent                : the multi-turn om2w agent harness (sab_science_sft_vllm.yaml)
#                          that writes final_script.py, then collect_sab_preds.py.
GEN_MODE="${GEN_MODE:-singleturn}"
echo "[eval] SAB inference mode=$GEN_MODE endpoint=$ENDPOINT model=$MODEL_NAME"
echo "[eval] tasks=$TASKS_FILE limit=$LIMIT workers=$WORKERS output=$OUTPUT_DIR"

if [[ "$GEN_MODE" == "singleturn" ]]; then
  # Optional --system-prompt-file via SYSTEM_PROMPT_FILE; defaults to the training
  # SYSTEM_PROMPT imported from make_d3gym_science_sft.py.
  SPF_ARG=()
  [[ -n "${SYSTEM_PROMPT_FILE:-}" ]] && SPF_ARG=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
  TEMPLATE_ARG=()
  [[ -n "${CHAT_TEMPLATE:-}" ]] && TEMPLATE_ARG=(--chat-template-file "$CHAT_TEMPLATE")
  TID_ARG=()
  [[ -n "${TASK_IDS:-}" ]] && TID_ARG=(--task-ids ${TASK_IDS})
  "$PY" scripts/sab_infer_singleturn.py \
    --endpoint "$ENDPOINT" \
    --model "$MODEL_NAME" \
    --tasks-file "$TASKS_FILE" \
    --pred-out "$PRED_OUT" \
    --run-log "$RUN_LOG" \
    --limit "$LIMIT" \
    --workers "$WORKERS" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --max-context-tokens "$MAX_MODEL_LEN" \
    --tokenizer-name-or-path "$CKPT" \
    "${SPF_ARG[@]}" "${TEMPLATE_ARG[@]}" "${TID_ARG[@]}" "$@"
else
  "$PY" -m miniswewebagent.run.benchmarks.om2w \
    -c mini.yaml -c "$BENCHMARK_CONFIG" \
    -c "model.endpoint=$ENDPOINT" \
    -c "model.model_name=$MODEL_NAME" \
    -c "model.max_output_tokens=$MAX_OUTPUT_TOKENS" \
    -c "environment.env.OPENAI_COMPATIBLE_ENDPOINT=$ENDPOINT" \
    -c "environment.env.OPENAI_COMPATIBLE_MODEL=$MODEL_NAME" \
    -c "environment.env.OPENAI_COMPATIBLE_API_KEY=dummy" \
    -c "environment.seed_symlinks.benchmark/datasets=$SAB_BENCHMARK_DATASETS" \
    --tasks-file "$TASKS_FILE" \
    --task-level "$TASK_LEVEL" \
    --limit "$LIMIT" \
    --workers "$WORKERS" \
    --no-evaluate \
    --output-dir "$OUTPUT_DIR" \
    "$@"
  echo "[collect] final_script.py -> pred_programs"
  "$PY" scripts/collect_sab_preds.py \
    --batch-dir "$OUTPUT_DIR" \
    --tasks-file "$TASKS_FILE" \
    --pred-out "$PRED_OUT" \
    --run-log "$RUN_LOG"
fi

cat <<EOF

============================================================
Stage A (inference) done.
  trajectories : $OUTPUT_DIR
  pred_programs: $PRED_OUT
  run log      : $RUN_LOG

Stage B (SAB scoring) — run inside the ScienceAgentBench repo:

  cd $SAB_REPO
  # dockerized (recommended; needs docker + 'pip install docker python-dotenv').
  # --split verified is REQUIRED (dataset only has the 'verified' split).
  # --openai_api_key is validated at startup; a placeholder works for CSV/JSON
  # tasks (only .png-output eval scripts actually call the visual judge).
  PYTHONPATH=. python -m evaluation.harness.run_evaluation \\
      --benchmark_path benchmark \\
      --pred_program_path $PRED_OUT \\
      --log_fname ${MODEL_NAME}_${STAMP}_eval.jsonl \\
      --run_id ${MODEL_NAME}_${STAMP} \\
      --split verified \\
      --openai_api_key \${OPENAI_API_KEY:-sk-placeholder} \\
      --max_workers 8

  # then metrics (note: --log_fname jsonl has all 102 rows; for a subset run,
  # read per-instance rows at index=instance_id-1 rather than the aggregate):
  python calculate_metrics.py \\
      --run_logs $RUN_LOG \\
      --eval_logs ${MODEL_NAME}_${STAMP}_eval.jsonl
============================================================
EOF
