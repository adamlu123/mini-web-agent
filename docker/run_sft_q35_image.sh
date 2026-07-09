#!/usr/bin/env bash
# In-pod driver for a LlamaFactory full-SFT run on the generic qwen3.5 image
#   aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415
#
# Much lighter bootstrap than the RL drivers: SFT needs no vllm rollout / skyrl /
# browserbase, only LlamaFactory + a few deps the image lacks. The image already
# bakes torch / transformers / deepspeed / accelerate. We add LlamaFactory
# (editable) + its pinned trl==0.24.0 + deepspeed's hjson/py-cpuinfo, then run
# `llamafactory-cli train $SFT_CONFIG` under torchrun on all GPUs.
#
# Required env (forwarded by submit_sft_q35_image.sh via --extra-env-vars):
#   SFT_CONFIG  -- yaml path relative to LlamaFactory/ (e.g. examples/train_full/...yaml)
#   NPROC       -- GPUs per node (torchrun --nproc_per_node)
# Auto-injected by submit_job.sh: PVC_MOUNT, USER_ALIAS, JOB_NAME
# Secret volume (HF token + HF_HOME cache live here):
#   /run/secrets/echo-rl-creds/cred.sh

set -e

echo "[boot] q35-image SFT pod $JOB_NAME on $(hostname)"
echo "[boot] SFT_CONFIG=${SFT_CONFIG:?SFT_CONFIG not set}  NPROC=${NPROC:?NPROC not set}"

# === multi-node wiring ========================================================
# The volcano `pytorch` plugin injects MASTER_ADDR/MASTER_PORT plus WORLD_SIZE
# (total nodes/pods) and RANK (this pod's node rank). LlamaFactory's launcher,
# however, reads NNODES/NODE_RANK, so map them here. Single-node jobs get
# WORLD_SIZE=1 / RANK=0 -> NNODES=1 / NODE_RANK=0 (identical to before).
export NNODES="${WORLD_SIZE:-1}"
export NODE_RANK="${RANK:-0}"
IS_MASTER=0; [ "$NODE_RANK" = "0" ] && IS_MASTER=1
echo "[boot] multi-node: NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-?} IS_MASTER=$IS_MASTER"

CODE_ROOT=$PVC_MOUNT/$USER_ALIAS/code
UPLOAD_ROOT=$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME
OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
LF_DIR=$CODE_ROOT/mini-web-agent/LlamaFactory
mkdir -p "$CODE_ROOT" "$OUTPUT_DIR"

# Per-job sentinel so workers don't rsync the shared $CODE_ROOT concurrently
# with the master (the dir lives on the shared PVC, seen by every pod).
SYNC_SENTINEL="$OUTPUT_DIR/.code_synced"

if ! command -v rsync >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq rsync
fi

if [ "$IS_MASTER" = "1" ]; then
  echo '[boot] === [master] copy uploaded code to stable PVC path ==='
  rm -f "$SYNC_SENTINEL"
  rsync -a --delete --no-perms --no-owner --no-group --no-times \
      --exclude 'LlamaFactory/saves/' \
      "$UPLOAD_ROOT/mini-web-agent/" "$CODE_ROOT/mini-web-agent/"
  # NOTE: --exclude 'LlamaFactory/saves/' keeps --delete from wiping a PRIOR run's
  # checkpoints (saves/ is gitignored -> not in the upload -> would otherwise be
  # deleted). Final ckpts are also copied to $PVC/.../models/ below for safety.

  # --- optional warm restart (RESUME_FROM_CKPT=<checkpoint-N dir>) -----------
  # save_only_model ckpts can't truly resume; prepare_warm_restart.py backs the
  # ckpt up to $PVC/.../models/, vision-merges it, and derives a continuation
  # yaml (remaining epochs + LR resumed from the ckpt's trainer_state.json).
  # Generated BEFORE the sentinel so worker ranks pick the same yaml up from the
  # shared $CODE_ROOT. TARGET_TOTAL_EPOCHS raises the total-epoch target (e.g.
  # original 3 + one more => 4). NOTE: submit with the SAME NODES as the
  # original run -- the epoch/LR math assumes an unchanged global batch.
  if [ -n "${RESUME_FROM_CKPT:-}" ]; then
    export HF_HOME="${HF_HOME:-$PVC_MOUNT/$USER_ALIAS/hf_cache}"
    python "$CODE_ROOT/mini-web-agent/docker/prepare_warm_restart.py" \
      --ckpt "$RESUME_FROM_CKPT" \
      --config "$LF_DIR/$SFT_CONFIG" \
      --out-config "$LF_DIR/examples/train_full/autogen_warm_restart_${JOB_NAME}.yaml" \
      --backup-root "$PVC_MOUNT/$USER_ALIAS/models" \
      --merge-script "$CODE_ROOT/mini-web-agent/scripts/merge_vision_from_base.py" \
      --hf-home "$HF_HOME" \
      --target-total-epochs "${TARGET_TOTAL_EPOCHS:-0}"
  fi
  touch "$SYNC_SENTINEL"
else
  # Warm-restart prep (ckpt backup copy + vision merge) runs on the master
  # before the sentinel and can take a while -> wait up to 60 min.
  echo "[boot] === [worker rank $NODE_RANK] waiting for master to sync code to $CODE_ROOT ==="
  for _ in $(seq 1 720); do [ -f "$SYNC_SENTINEL" ] && break; sleep 5; done
  [ -f "$SYNC_SENTINEL" ] || { echo "[boot][error] timed out waiting for master code sync"; exit 1; }
  echo "[boot] === [worker rank $NODE_RANK] code sync detected; continuing ==="
fi

# Every rank trains the derived continuation yaml when warm-restarting (the
# master just generated it at this shared path).
if [ -n "${RESUME_FROM_CKPT:-}" ]; then
  SFT_CONFIG="examples/train_full/autogen_warm_restart_${JOB_NAME}.yaml"
  echo "[boot] warm restart -> SFT_CONFIG=$SFT_CONFIG"
fi

echo '[boot] === install LlamaFactory + the few deps the image lacks (--no-deps) ==='
echo "[boot] python -> $(command -v python) ; $(python -V 2>&1)"
# LlamaFactory's pyproject uses the hatchling build backend; with
# --no-build-isolation pip needs it already present in the env, but the image
# doesn't bake it -> editable install dies with `Cannot import 'hatchling.build'`.
# Install the (pure-python) build backend + editables first. NOT --no-deps:
# hatchling itself needs pathspec/pluggy/packaging/trove-classifiers, and those
# are pure-python build-time deps that don't touch the image's torch stack.
# Version comes from src/llamafactory/extras/env.py, so no git/VCS at build time.
pip install hatchling editables
pip install --no-deps --no-build-isolation -e "$LF_DIR"
# metrics extras (nltk / jieba / rouge-chinese) for eval/compute_metrics paths.
pip install --no-deps -r "$LF_DIR/requirements/metrics.txt"
# peft: image's may be too old/mismatched for this LlamaFactory; force a fresh
# wheel without touching the rest of the (image-owned) torch stack.
pip install --no-deps peft
# LlamaFactory 0.9.x imports BOTH trl.AutoModelForCausalLMWithValueHead AND
# trl.models.utils.prepare_deepspeed -> only trl 0.18-0.24 has both (it pins
# trl>=0.18,<=0.24). The image may ship a newer trl; force 0.24.0.
pip install --no-deps "trl==0.24.0"
# deepspeed config parsing needs hjson; cpuinfo for its launcher.
python -c "import hjson" 2>/dev/null || pip install --no-deps hjson
python -c "import cpuinfo" 2>/dev/null || pip install --no-deps py-cpuinfo

echo '[boot] === source creds (HF token + HF_HOME cache on PVC) ==='
[ -f /run/secrets/echo-rl-creds/cred.sh ] && source /run/secrets/echo-rl-creds/cred.sh
# Cache HF models on the PVC so re-runs don't re-download Qwen3.5.
export HF_HOME="${HF_HOME:-$PVC_MOUNT/$USER_ALIAS/hf_cache}"
mkdir -p "$HF_HOME"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# submit_job.sh injects NCCL_DEBUG=INFO, which floods the log with thousands of
# `NCCL INFO Channel ...` lines and buries the training loss/progress. Drop it to
# WARN so only real NCCL problems show. Override with NCCL_DEBUG_OVERRIDE=INFO.
export NCCL_DEBUG="${NCCL_DEBUG_OVERRIDE:-WARN}"
# Qwen3.5 full SFT on B200 hit intermittent NCCL/CUDA peer-memory failures in
# the NVLink path after a few hundred steps. Prefer the slower host/shared-memory
# fallback over direct GPU peer access unless explicitly overridden.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"

echo '[boot] === pre-flight ==='
nvidia-smi -L
python - <<'PY'
import torch, transformers, deepspeed, accelerate, peft, datasets, trl
from trl import AutoModelForCausalLMWithValueHead  # noqa
from trl.models.utils import prepare_deepspeed  # noqa
import llamafactory
from transformers import AutoConfig
AutoConfig.for_model("qwen3_5")  # arch must be known
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| trl", trl.__version__, "| deepspeed", deepspeed.__version__,
      "| llamafactory", llamafactory.__version__)
print("SFT pre-flight OK (qwen3_5 arch supported)")
PY

push_ckpt_to_blob() {
  src_dir="$1"
  blob_name="$2"
  account="${AZBLOB_ACCOUNT_NAME:-aifrontiers}"
  container="${AZBLOB_CONTAINER_NAME:-data}"
  alias="${USER_ALIAS:-${USER%@*}}"
  prefix="${AZBLOB_PREFIX:-bonete/ckpts/${alias}}"
  dest="https://${account}.blob.core.windows.net/${container}/${prefix}/${blob_name}"

  command -v az >/dev/null 2>&1 || {
    echo "[blob] az CLI not found; installing..."
    command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash >/dev/null 2>&1
  }
  command -v az >/dev/null 2>&1 || { echo "[blob][error] az install failed"; return 1; }

  echo "[blob] push: ${src_dir%/}/* -> ${dest}"
  if [ -n "${AZBLOB_SAS_TOKEN:-}" ]; then
    az storage copy -s "${src_dir%/}/*" -d "${dest}?${AZBLOB_SAS_TOKEN#\?}" --recursive
  else
    if ! az account show >/dev/null 2>&1; then
      if [ -n "${AZURE_TENANT_ID:-}" ] && [ -n "${AZURE_CLIENT_ID:-}" ] && [ -r "${AZURE_FEDERATED_TOKEN_FILE:-/nonexistent}" ]; then
        echo "[blob] az login via workload-identity (client=$AZURE_CLIENT_ID)"
        az login --service-principal --tenant "$AZURE_TENANT_ID" --username "$AZURE_CLIENT_ID" \
          --federated-token "$(cat "$AZURE_FEDERATED_TOKEN_FILE")" --allow-no-subscriptions --output none
      else
        echo "[blob][error] no auth: set AZBLOB_SAS_TOKEN or provide workload-identity env"
        return 1
      fi
    fi
    az storage copy -s "${src_dir%/}/*" -d "$dest" --recursive --auth-mode login
  fi
}

echo "[boot] === launching SFT: $SFT_CONFIG on $NPROC GPUs ==="
cd "$LF_DIR"
RC=0
FORCE_TORCHRUN=1 NPROC_PER_NODE="$NPROC" DISABLE_VERSION_CHECK=1 \
  llamafactory-cli train "$SFT_CONFIG" 2>&1 | tee -a "$OUTPUT_DIR/console.log"
RC=${PIPESTATUS[0]}
echo "[boot] SFT exited rc=$RC"

# === sync the final checkpoint out of the volatile code dir ===================
# $CODE_ROOT is rsync --delete'd at the START of every job, so a ckpt left under
# LlamaFactory/saves/ would be WIPED by the next run. On success, copy the final
# HF model (output_dir root = config + safetensors + tokenizer) to a stable
# per-model dir on the PVC that survives future jobs. Override dest via
# SYNC_CKPT_DIR; disable entirely with SYNC_CKPT=0.
# Only the master node post-processes (sync + vision-merge). Workers' torchrun
# procs have already exited; they must NOT race on the shared ckpt dir.
# EXCEPTION: with EVAL_AFTER=1 + the (default) harness eval backend, workers
# do NOT exit -- they wait for the master's ckpt-ready sentinel and then run
# their own data-parallel eval shard (docker/run_dist_eval_q35_image.sh), so the
# whole job's GPUs eval in parallel instead of idling while the master evals.
EVAL_READY="$OUTPUT_DIR/.ckpt_ready_for_eval"
if [ "$IS_MASTER" != "1" ]; then
  if [ "$RC" -eq 0 ] && [ "${EVAL_AFTER:-0}" = "1" ] && [ "${EVAL_BACKEND:-harness}" = "harness" ]; then
    # Full-saves sync can be slow (up to hundreds of GB without SYNC_FINAL_ONLY):
    # default wait 3h, override with EVAL_READY_TIMEOUT (seconds).
    echo "[boot] [worker rank $NODE_RANK] training done; waiting for master ckpt sync+merge ($EVAL_READY)"
    for _ in $(seq 1 $(( ${EVAL_READY_TIMEOUT:-10800} / 10 ))); do [ -f "$EVAL_READY" ] && break; sleep 10; done
    if [ -f "$EVAL_READY" ]; then
      ERC=0
      EVAL_CKPT="$(cat "$EVAL_READY")" \
      EVAL_RUN_ID="${EVAL_RUN_ID:-$JOB_NAME}" \
      EVAL_NNODES="$NNODES" EVAL_NODE_RANK="$NODE_RANK" \
      REPO="$CODE_ROOT/mini-web-agent" \
        bash "$CODE_ROOT/mini-web-agent/docker/run_dist_eval_q35_image.sh" || ERC=$?
      echo "[boot] [worker rank $NODE_RANK] eval shard exited rc=$ERC"
      exit "$ERC"
    fi
    echo "[boot][error] [worker rank $NODE_RANK] timed out waiting for $EVAL_READY; no eval shard run"
    exit 1
  fi
  echo "[boot] [worker rank $NODE_RANK] training done rc=$RC; skipping sync/eval (master handles it)"
  exit "$RC"
fi

if [ "$RC" -eq 0 ] && [ "${SYNC_CKPT:-1}" = "1" ]; then
  CKPT_REL=$(grep -E '^[[:space:]]*output_dir:' "$LF_DIR/$SFT_CONFIG" | head -1 | sed 's/#.*//' | awk '{print $2}')
  if [ -n "$CKPT_REL" ] && [ -d "$LF_DIR/$CKPT_REL" ]; then
    SYNC_CKPT_DIR="${SYNC_CKPT_DIR:-$PVC_MOUNT/$USER_ALIAS/models/${CKPT_REL#saves/}}"
    echo "[sync] final ckpt: $LF_DIR/$CKPT_REL -> $SYNC_CKPT_DIR"
    mkdir -p "$SYNC_CKPT_DIR"
    # SYNC_FINAL_ONLY=1 -> sync only the final model at the output_dir root;
    # intermediate checkpoint-* dirs (save_steps snapshots, ~18GB each for 9B)
    # are left behind in the volatile code dir and wiped by the next job.
    SYNC_EXCLUDE=""
    if [ "${SYNC_FINAL_ONLY:-0}" = "1" ]; then
      SYNC_EXCLUDE="--exclude=checkpoint-*"
      echo "[sync] SYNC_FINAL_ONLY=1 -> intermediate checkpoint-* dirs NOT synced"
    fi
    if rsync -a --delete $SYNC_EXCLUDE "$LF_DIR/$CKPT_REL"/ "$SYNC_CKPT_DIR"/; then
      echo "[sync] OK -- stable ckpt path (survives future jobs): $SYNC_CKPT_DIR"
      # Complete the VL ckpt: LlamaFactory text-SFT on qwen3_5 drops the vision
      # tower from the saved weights (its registered vision keys are mis-prefixed
      # 'visual.*' vs the real 'model.visual.*'), so the ckpt can't reload as
      # Qwen3_5ForConditionalGeneration. Merge the (unchanged) vision tower back
      # from the base in HF_HOME so every saved ckpt loads standalone in vLLM.
      # Disable with MERGE_VISION=0.
      if [ "${MERGE_VISION:-1}" = "1" ]; then
        MODEL_ID=$(grep -E '^[[:space:]]*model_name_or_path:' "$LF_DIR/$SFT_CONFIG" | head -1 | sed 's/#.*//' | awk '{print $2}')
        BASE_DIR=$(ls -d "$HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots/"*/ 2>/dev/null | head -1)
        MERGE_PY="$CODE_ROOT/mini-web-agent/scripts/merge_vision_from_base.py"
        if [ -n "$BASE_DIR" ] && [ -f "$MERGE_PY" ]; then
          echo "[merge] completing VL ckpt from base: $BASE_DIR"
          python "$MERGE_PY" --ckpt "$SYNC_CKPT_DIR" --base "$BASE_DIR" \
            || echo "[merge][warn] vision merge failed; ckpt may lack the vision tower"
        else
          echo "[merge][warn] base ('$BASE_DIR') or merge script not found; skipping vision merge"
        fi
      fi
      # Auto-upload to Azure Blob so a dev box can pull the finished ckpt without
      # needing a live pod. Auth = workload-identity if the federated-token env is
      # present, else the injected AZBLOB_SAS_TOKEN. Disable with AZBLOB_AUTO_PUSH=0.
      if [ "${AZBLOB_AUTO_PUSH:-1}" = "1" ]; then
        echo "[sync] AZBLOB_AUTO_PUSH=${AZBLOB_AUTO_PUSH:-1} -> uploading ckpt to blob"
        if push_ckpt_to_blob "$SYNC_CKPT_DIR" "${CKPT_REL#saves/}"; then
          echo "[sync] blob upload OK. pull on a dev box: bash scripts/az_ckpt.sh pull ${CKPT_REL#saves/} <dest>"
        else
          echo "[sync][warn] blob upload failed (auth/SAS?); ckpt still on PVC at $SYNC_CKPT_DIR"
        fi
      else
        echo "[sync] AZBLOB_AUTO_PUSH=0; ckpt only on PVC. pull via a live pod or rerun upload manually: bash scripts/az_ckpt.sh push $SYNC_CKPT_DIR ${CKPT_REL#saves/}"
      fi
    else
      echo "[sync][warn] rsync failed; ckpt remains (volatile) at $LF_DIR/$CKPT_REL"
    fi
  else
    echo "[sync][warn] output_dir '$CKPT_REL' not found under $LF_DIR; nothing synced"
  fi
fi

# === optional: chain a cluster EVAL on the freshly-trained ckpt ===============
# When EVAL_AFTER=1 (set by docker/submit_sft_eval_q35_image.sh), evaluate the
# trained HF ckpt right here in the SAME job, so one submission does train->eval.
# EVAL_BACKEND selects the stack:
#   harness (default) -- mini-web-agent's own OM2W harness (vllm serve + agent
#     loop, docker/run_dist_eval_q35_image.sh). Data-parallel across ALL the
#     job's nodes: the master publishes the ckpt path via $EVAL_READY, every
#     rank (master included) runs its own shard, the master then judges the
#     merged outputs. Resumable: re-running with the same EVAL_RUN_ID skips
#     finished tasks.
#   skyrl -- the legacy SkyRL eval_entrypoint (master node only; needs the
#     SkyRL upload + echo-rl secret volumes from the combined submit).
if [ "$RC" -eq 0 ] && [ "${EVAL_AFTER:-0}" = "1" ]; then
  # Resolve the HF ckpt to evaluate: prefer the stable synced dir, else the
  # in-place saves dir under LlamaFactory.
  EVAL_CKPT_PATH="${SYNC_CKPT_DIR:-}"
  if [ -z "$EVAL_CKPT_PATH" ] || [ ! -d "$EVAL_CKPT_PATH" ]; then
    CKPT_REL=$(grep -E '^[[:space:]]*output_dir:' "$LF_DIR/$SFT_CONFIG" | head -1 | sed 's/#.*//' | awk '{print $2}')
    EVAL_CKPT_PATH="$LF_DIR/$CKPT_REL"
  fi
  if [ "${EVAL_BACKEND:-harness}" = "harness" ]; then
    EVAL_DRIVER="$CODE_ROOT/mini-web-agent/docker/run_dist_eval_q35_image.sh"
    if [ -d "$EVAL_CKPT_PATH" ] && [ -f "$EVAL_DRIVER" ]; then
      echo "[eval] === EVAL_AFTER=1 (harness) -> $NNODES-shard eval on trained ckpt: $EVAL_CKPT_PATH ==="
      # Unblock the waiting worker ranks, then run OUR shard (rank 0) + judge.
      echo "$EVAL_CKPT_PATH" > "$EVAL_READY"
      ERC=0
      EVAL_CKPT="$EVAL_CKPT_PATH" \
      EVAL_RUN_ID="${EVAL_RUN_ID:-$JOB_NAME}" \
      EVAL_NNODES="$NNODES" EVAL_NODE_RANK=0 \
      REPO="$CODE_ROOT/mini-web-agent" \
        bash "$EVAL_DRIVER" || ERC=$?
      echo "[eval] harness eval exited rc=$ERC (train rc was $RC)"
      [ "$ERC" -ne 0 ] && RC=$ERC
    else
      echo "[eval][warn] skipping eval: ckpt ('$EVAL_CKPT_PATH') or driver ('$EVAL_DRIVER') missing"
      # Don't leave the worker ranks hanging on the sentinel.
      echo "" > "$EVAL_READY"
    fi
  else
    EVAL_DRIVER="$CODE_ROOT/mini-web-agent/docker/run_eval_q35_image.sh"
    if [ -d "$EVAL_CKPT_PATH" ] && [ -f "$EVAL_DRIVER" ]; then
      echo "[eval] === EVAL_AFTER=1 (skyrl) -> cluster eval on trained ckpt: $EVAL_CKPT_PATH ==="
      # The SFT phase already rsynced mini-web-agent; the eval driver still needs
      # to rsync SkyRL + bootstrap the RL/eval stack on top of the LlamaFactory env.
      ERC=0
      EVAL_CKPT="$EVAL_CKPT_PATH" \
      EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen35_9b_web_agent_easy_eval_sft.yaml}" \
      EVAL_RUN_TAG="${EVAL_RUN_TAG:-merged}" \
        bash "$EVAL_DRIVER" || ERC=$?
      echo "[eval] cluster eval exited rc=$ERC (train rc was $RC)"
      # Surface an eval failure in the job status but don't pretend training failed.
      [ "$ERC" -ne 0 ] && RC=$ERC
    else
      echo "[eval][warn] skipping eval: ckpt ('$EVAL_CKPT_PATH') or driver ('$EVAL_DRIVER') missing"
    fi
  fi
fi
exit "$RC"
