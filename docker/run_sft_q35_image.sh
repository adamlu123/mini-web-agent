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

CODE_ROOT=$PVC_MOUNT/$USER_ALIAS/code
UPLOAD_ROOT=$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME
OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
LF_DIR=$CODE_ROOT/mini-web-agent/LlamaFactory
mkdir -p "$CODE_ROOT" "$OUTPUT_DIR"

echo '[boot] === copy uploaded code to stable PVC path ==='
if ! command -v rsync >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq rsync
fi
rsync -a --delete --no-perms --no-owner --no-group --no-times \
    --exclude 'LlamaFactory/saves/' \
    "$UPLOAD_ROOT/mini-web-agent/" "$CODE_ROOT/mini-web-agent/"
# NOTE: --exclude 'LlamaFactory/saves/' keeps --delete from wiping a PRIOR run's
# checkpoints (saves/ is gitignored -> not in the upload -> would otherwise be
# deleted). Final ckpts are also copied to $PVC/.../models/ below for safety.

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
if [ "$RC" -eq 0 ] && [ "${SYNC_CKPT:-1}" = "1" ]; then
  CKPT_REL=$(grep -E '^[[:space:]]*output_dir:' "$LF_DIR/$SFT_CONFIG" | head -1 | sed 's/#.*//' | awk '{print $2}')
  if [ -n "$CKPT_REL" ] && [ -d "$LF_DIR/$CKPT_REL" ]; then
    SYNC_CKPT_DIR="${SYNC_CKPT_DIR:-$PVC_MOUNT/$USER_ALIAS/models/${CKPT_REL#saves/}}"
    echo "[sync] final ckpt: $LF_DIR/$CKPT_REL -> $SYNC_CKPT_DIR"
    mkdir -p "$SYNC_CKPT_DIR"
    if rsync -a --delete "$LF_DIR/$CKPT_REL"/ "$SYNC_CKPT_DIR"/; then
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
      # Optional: auto-upload to Azure Blob so a dev box can pull it (works even
      # after a pod reschedule / node change). Auth = workload-identity if the
      # federated-token env is present, else the injected AZBLOB_SAS_TOKEN.
      if [ "${AZBLOB_AUTO_PUSH:-0}" = "1" ]; then
        AZ="$CODE_ROOT/mini-web-agent/scripts/az_ckpt.sh"
        echo "[sync] AZBLOB_AUTO_PUSH=1 -> uploading ckpt to blob"
        if bash "$AZ" push "$SYNC_CKPT_DIR" "${CKPT_REL#saves/}"; then
          echo "[sync] blob upload OK. pull on a dev box: bash scripts/az_ckpt.sh pull ${CKPT_REL#saves/} <dest>"
        else
          echo "[sync][warn] blob upload failed (auth/SAS?); ckpt still on PVC at $SYNC_CKPT_DIR"
        fi
      else
        echo "[sync] pull to a dev box: bash scripts/az_ckpt.sh pull ${CKPT_REL#saves/} <dest>  (or set AZBLOB_AUTO_PUSH=1 to upload automatically)"
      fi
    else
      echo "[sync][warn] rsync failed; ckpt remains (volatile) at $LF_DIR/$CKPT_REL"
    fi
  else
    echo "[sync][warn] output_dir '$CKPT_REL' not found under $LF_DIR; nothing synced"
  fi
fi

# === optional: chain a cluster EVAL on the freshly-trained ckpt ===============
# When EVAL_AFTER=1 (set by docker/submit_sft_eval_q35_image.sh), run the
# cluster eval driver on the trained HF ckpt right here in the SAME job/pod, so
# one submission does train -> eval. The eval needs the SkyRL + RL/eval stack
# (uploaded alongside mini-web-agent by the combined submit). Disabled by
# default -> the plain SFT submit behaves exactly as before.
if [ "$RC" -eq 0 ] && [ "${EVAL_AFTER:-0}" = "1" ]; then
  # Resolve the HF ckpt to evaluate: prefer the stable synced dir, else the
  # in-place saves dir under LlamaFactory.
  EVAL_CKPT_PATH="${SYNC_CKPT_DIR:-}"
  if [ -z "$EVAL_CKPT_PATH" ] || [ ! -d "$EVAL_CKPT_PATH" ]; then
    CKPT_REL=$(grep -E '^[[:space:]]*output_dir:' "$LF_DIR/$SFT_CONFIG" | head -1 | sed 's/#.*//' | awk '{print $2}')
    EVAL_CKPT_PATH="$LF_DIR/$CKPT_REL"
  fi
  EVAL_DRIVER="$CODE_ROOT/mini-web-agent/docker/run_eval_q35_image.sh"
  if [ -d "$EVAL_CKPT_PATH" ] && [ -f "$EVAL_DRIVER" ]; then
    echo "[eval] === EVAL_AFTER=1 -> cluster eval on trained ckpt: $EVAL_CKPT_PATH ==="
    # The SFT phase already rsynced mini-web-agent; the eval driver still needs
    # to rsync SkyRL + bootstrap the RL/eval stack on top of the LlamaFactory env.
    EVAL_CKPT="$EVAL_CKPT_PATH" \
    EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen35_9b_web_agent_easy_eval_sft.yaml}" \
    EVAL_RUN_TAG="${EVAL_RUN_TAG:-merged}" \
      bash "$EVAL_DRIVER"
    ERC=$?
    echo "[eval] cluster eval exited rc=$ERC (train rc was $RC)"
    # Surface an eval failure in the job status but don't pretend training failed.
    [ "$ERC" -ne 0 ] && RC=$ERC
  else
    echo "[eval][warn] skipping eval: ckpt ('$EVAL_CKPT_PATH') or driver ('$EVAL_DRIVER') missing"
  fi
fi
exit "$RC"
