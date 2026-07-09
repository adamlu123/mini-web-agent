#!/usr/bin/env bash
# Submit a COMBINED "train then eval" job on the generic qwen3.5 image, runs to
# completion:
#   1. LlamaFactory full-SFT (docker/run_sft_q35_image.sh) on $SFT_CONFIG, then
#   2. on success, an eval of the trained HF ckpt. EVAL_BACKEND picks the stack:
#      - harness (default): mini-web-agent's own OM2W harness
#        (docker/run_dist_eval_q35_image.sh) -- every node serves the ckpt with
#        a local vLLM (tp=8) and runs 1/NODES of the tasks; the master judges
#        the merged outputs at the end. Multi-node train => multi-node eval,
#        no idle GPUs. Resumable via EVAL_RUN_ID.
#      - skyrl: the legacy SkyRL eval_entrypoint (master node only; uploads
#        SkyRL + mounts the echo-rl-openai judge secret).
#
# Usage:
#   NODES=2 SFT_CONFIG=examples/train_full/....yaml bash docker/submit_sft_eval_q35_image.sh
#   TASK_LEVEL=easy TOTAL_WORKERS=80 NODES=4 SFT_CONFIG=... bash docker/submit_sft_eval_q35_image.sh
#
# Kill manually with:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415}"
GPUS="${GPUS:-8}"
# Multi-node: train data-parallel on all nodes; with the harness eval backend
# the SAME nodes then eval data-parallel (1/NODES of the tasks each).
NODES="${NODES:-1}"
# Train config: path relative to LlamaFactory/ (lives inside the uploaded repo).
SFT_CONFIG="${SFT_CONFIG:-examples/train_full/qwen35_9b_websft_merged.yaml}"

# --- eval backend selection ---------------------------------------------------
EVAL_BACKEND="${EVAL_BACKEND:-harness}"

# harness-backend knobs (mini-web-agent OM2W harness; see run_dist_eval_q35_image.sh)
TASK_LEVEL="${TASK_LEVEL:-all}"          # easy / medium / hard / all (300 tasks)
LIMIT="${LIMIT:-0}"
TOTAL_WORKERS="${TOTAL_WORKERS:-80}"     # total browserbase sessions across ALL nodes
WORKERS="${WORKERS:-$(( (TOTAL_WORKERS + NODES - 1) / NODES ))}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
RETRY_FAILED="${RETRY_FAILED:-0}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/data/t-yifeili/webchain_sampling/cred.sh}"
CREDENTIALS_SECRET="${CREDENTIALS_SECRET:-t-yifeili-webchain-sampling-creds}"

# skyrl-backend knobs (legacy; only used when EVAL_BACKEND=skyrl)
EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen35_9b_web_agent_all3_eval_sft.yaml}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-merged_9b}"
EVAL_AGENT_CONCURRENCY="${EVAL_AGENT_CONCURRENCY:-80}"
EVAL_NUM_ENGINES="${EVAL_NUM_ENGINES:-8}"
EVAL_EXEC_BACKEND="${EVAL_EXEC_BACKEND:-mp}"

# LIGHT_UPLOAD=1 (default): upload a code tree without LlamaFactory/data/web_agent_*;
# training data must live at /mnt/pvc/experiments/t-yifeili/data/<bundle> (see
# docker/upload_data_to_pvc.sh) with the yaml's dataset_dir/media_dir pointing there.
LIGHT_UPLOAD="${LIGHT_UPLOAD:-1}"

[[ -d "$MINI_WEB_AGENT_DIR" ]] || { echo "[error] missing $MINI_WEB_AGENT_DIR"; exit 1; }
[[ -f "$MINI_WEB_AGENT_DIR/LlamaFactory/$SFT_CONFIG" ]] || { echo "[error] SFT config not found: LlamaFactory/$SFT_CONFIG"; exit 1; }
if [[ "$EVAL_BACKEND" == "harness" ]]; then
  [[ -f "$CREDENTIALS_FILE" ]] || { echo "[error] credentials file not found: $CREDENTIALS_FILE"; exit 1; }
else
  [[ -d "$SKYRL_DIR" ]] || { echo "[error] missing $SKYRL_DIR"; exit 1; }
  [[ -f "$MINI_WEB_AGENT_DIR/$EVAL_CONFIG" ]] || { echo "[error] eval config not found: $EVAL_CONFIG"; exit 1; }
fi

if [[ "$LIGHT_UPLOAD" == "1" ]]; then
    MINI_WEB_AGENT_DIR="$(SRC="$MINI_WEB_AGENT_DIR" bash "$(dirname "$0")/make_light_code_tree.sh" | tail -1)"
    echo "[submit_sft_eval_q35_image] LIGHT_UPLOAD=1 -> uploading $MINI_WEB_AGENT_DIR"
fi

export PATH="$HOME/.krew/bin:$PATH"
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"
# wandb key 预检:没 key 自动加载共享 key,加载不到直接拒绝提交(防 401 打挂 job)
source "$(dirname "$0")/wandb_key_preflight.sh"
export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"
export NAMESPACE="${NAMESPACE:-bonete61}"

echo "[submit_sft_eval_q35_image] NODES=$NODES GPUS=$GPUS (total $((NODES*GPUS)) GPUs) IMAGE=$IMAGE"
echo "[submit_sft_eval_q35_image] SFT_CONFIG=$SFT_CONFIG"
if [[ "$EVAL_BACKEND" == "harness" ]]; then
  echo "[submit_sft_eval_q35_image] EVAL: harness, level=$TASK_LEVEL, ${WORKERS} workers/node (total ~$((WORKERS*NODES)) sessions), $NODES shard(s)"
else
  echo "[submit_sft_eval_q35_image] EVAL: skyrl, $EVAL_CONFIG (tag=$EVAL_RUN_TAG, workers=$EVAL_AGENT_CONCURRENCY, engines=$EVAL_NUM_ENGINES/$EVAL_EXEC_BACKEND)"
fi
echo "[submit_sft_eval_q35_image] PRIORITY=$PRIORITY CLASS=$PRIORITY_CLASS_NAME PROJECT=$PROJECT_NAME"

# Forward both phases' knobs. EVAL_AFTER=1 flips on the eval chaining inside
# run_sft_q35_image.sh after a successful train + ckpt sync. The SFT driver
# uploads the final ckpt to blob by default; set AZBLOB_AUTO_PUSH=0 to disable.
EXTRA_ENV="SFT_CONFIG=${SFT_CONFIG},NPROC=${GPUS},EVAL_AFTER=1,EVAL_BACKEND=${EVAL_BACKEND}"
# Warm restart: RESUME_FROM_CKPT=<checkpoint-N dir on PVC> (+ optional
# TARGET_TOTAL_EPOCHS). In-pod prep backs it up + vision-merges + derives the
# continuation yaml. Submit with the SAME NODES as the original run.
[[ -n "${RESUME_FROM_CKPT:-}" ]]    && EXTRA_ENV="${EXTRA_ENV},RESUME_FROM_CKPT=${RESUME_FROM_CKPT}"
[[ -n "${TARGET_TOTAL_EPOCHS:-}" ]] && EXTRA_ENV="${EXTRA_ENV},TARGET_TOTAL_EPOCHS=${TARGET_TOTAL_EPOCHS}"
if [[ "$EVAL_BACKEND" == "harness" ]]; then
  EXTRA_ENV="${EXTRA_ENV},TASK_LEVEL=${TASK_LEVEL},LIMIT=${LIMIT},WORKERS=${WORKERS}"
  EXTRA_ENV="${EXTRA_ENV},JUDGE_RUNS=${JUDGE_RUNS},JUDGE_NUM_PROC=${JUDGE_NUM_PROC},RETRY_FAILED=${RETRY_FAILED}"
  [[ -n "${EVAL_RUN_ID:-}" ]]         && EXTRA_ENV="${EXTRA_ENV},EVAL_RUN_ID=${EVAL_RUN_ID}"
  [[ -n "${MAX_CONTEXT_TOKENS:-}" ]]  && EXTRA_ENV="${EXTRA_ENV},MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS}"
  [[ -n "${MAX_MODEL_LEN:-}" ]]       && EXTRA_ENV="${EXTRA_ENV},MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  [[ -n "${BENCHMARK_CONFIG:-}" ]]    && EXTRA_ENV="${EXTRA_ENV},BENCHMARK_CONFIG=${BENCHMARK_CONFIG}"
else
  EXTRA_ENV="${EXTRA_ENV},EVAL_CONFIG=${EVAL_CONFIG},EVAL_RUN_TAG=${EVAL_RUN_TAG},EVAL_AGENT_CONCURRENCY=${EVAL_AGENT_CONCURRENCY},EVAL_NUM_ENGINES=${EVAL_NUM_ENGINES},EVAL_EXEC_BACKEND=${EVAL_EXEC_BACKEND}"
fi
[[ -n "${SYNC_FINAL_ONLY:-}" ]] && EXTRA_ENV="${EXTRA_ENV},SYNC_FINAL_ONLY=${SYNC_FINAL_ONLY}"
[[ -n "${AZBLOB_AUTO_PUSH:-}" ]] && EXTRA_ENV="${EXTRA_ENV},AZBLOB_AUTO_PUSH=${AZBLOB_AUTO_PUSH}"
[[ -n "${AZBLOB_SAS_TOKEN:-}" ]] && EXTRA_ENV="${EXTRA_ENV},AZBLOB_SAS_TOKEN=${AZBLOB_SAS_TOKEN}"
[[ -n "${AZBLOB_PREFIX:-}" ]]    && EXTRA_ENV="${EXTRA_ENV},AZBLOB_PREFIX=${AZBLOB_PREFIX}"

# Uploads + secret volumes depend on the eval backend:
#   harness -> mini-web-agent only; webchain secret (browserbase + judge key)
#   skyrl   -> + SkyRL upload; echo-rl-openai secret (judge key)
UPLOADS=( "$MINI_WEB_AGENT_DIR" )
SECRET_ARGS=( --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds )
if [[ "$EVAL_BACKEND" == "harness" ]]; then
  kubectl -n "$NAMESPACE" create secret generic "$CREDENTIALS_SECRET" \
    --from-file=cred.sh="$CREDENTIALS_FILE" \
    --dry-run=client -o yaml | kubectl -n "$NAMESPACE" apply -f -
  SECRET_ARGS+=( --secret-volume "$CREDENTIALS_SECRET:/run/secrets/webchain-sampling" )
else
  UPLOADS+=( "$SKYRL_DIR" )
  SECRET_ARGS+=( --secret-volume echo-rl-openai:/run/secrets/echo-rl-openai )
fi

# Tiny --cmd execs the uploaded SFT driver (which chains the eval driver when
# EVAL_AFTER=1). WAF-safe.
bash "$SUBMIT" \
    --upload "${UPLOADS[@]}" \
    --image "$IMAGE" \
    --node "$NODES" --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    "${SECRET_ARGS[@]}" \
    --extra-env-vars "$EXTRA_ENV" \
    --follow-logs \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_sft_q35_image.sh'
