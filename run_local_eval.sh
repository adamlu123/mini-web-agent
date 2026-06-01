#!/usr/bin/env bash
# Local (4xH100) eval-only driver for the web-agent.
#
# Loads the BASE Qwen3.5 HF weights into local vLLM, runs a single evaluate()
# pass over the om2w easy train + val parquets (each scored separately), and
# logs eval/<data_source>/... to the console (no wandb, no FSDP, no training).
#
# Same rollout/model knobs as training (max_turns, max_context_tokens,
# max_tokens_per_generation, max_prompt_length, eval_sampling_params, OSW judge,
# Browserbase env) -- only eval_n_samples_per_prompt is dropped to 1 and the
# logger is switched to console.
#
# Usage:
#   MODEL=4b bash run_local_eval.sh                 # base HF weights (or MODEL=9b)
#   MODEL=4b CKPT=/path/to/hf/global_step_9 \        # a trained, converted ckpt
#       bash run_local_eval.sh
#
# CKPT must point at a HuggingFace-format weights dir (safetensors + config +
# tokenizer). SkyRL/FSDP checkpoints under .../ckpts/global_step_N/policy are
# sharded DTensors -- convert them first with:
#   python -m echo_rl.web_agent.scripts.convert_fsdp_ckpt_to_hf \
#       --ckpt .../ckpts/global_step_N/policy --output .../hf/global_step_N
set -euo pipefail

MODEL="${MODEL:-4b}"
case "$MODEL" in
  4b) CONFIG="configs/qwen35_4b_web_agent_easy_eval.yaml" ;;
  9b) CONFIG="configs/qwen35_9b_web_agent_easy_eval.yaml" ;;
  *) echo "[error] MODEL must be 4b or 9b"; exit 1 ;;
esac

# Optional: override the policy weights with a trained checkpoint (HF format).
MODEL_PATH_OVERRIDE=()
if [[ -n "${CKPT:-}" ]]; then
  [[ -d "$CKPT" ]] || { echo "[error] CKPT dir not found: $CKPT"; exit 1; }
  MODEL_PATH_OVERRIDE=( "trainer.policy.model.path=${CKPT}" )
fi

REPO=/data/t-yifeili/mini-web-agent
PY=/data/t-yifeili/miniconda3/envs/echo-rl/bin/python

# --- creds: browserbase project id + HF token (api keys already in env) ---
# cred.sh exports BROWSERBASE_PROJECT_ID / BROWSERBASE_API_KEY / HF_TOKEN.
# We keep the working OPENAI_API_BACKUP_KEY (judge) that is already in the env;
# cred.sh's OPENAI_API_KEY is the dead phyagi token, so re-unset the gateway.
source /home/luyadong/cred.sh
unset OPENAI_GATEWAY_API_KEY || true
export OPENAI_GATEWAY_ENDPOINT=''

export MINI_WEB_AGENT_ROOT=$REPO
export ECHO_RL_DATA=$REPO/data/web_agent
# RUN_TAG distinguishes ckpt runs (e.g. RUN_TAG=step9) from base runs.
export OUTPUT_DIR=$REPO/eval_outputs/${MODEL}_easy_${RUN_TAG:+${RUN_TAG}_}$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "[run] MODEL=$MODEL CONFIG=$CONFIG"
echo "[run] WEIGHTS=${CKPT:-<base HF weights from config>}"
echo "[run] OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi -L

cd "$REPO"
# NOTE: colocate_all=false is REQUIRED for eval-only. With colocate_all=true
# (the training default) SkyRL sleeps the vLLM engine at level=2 right after
# startup (discards weights from VRAM, no CPU backup) expecting a later NCCL
# weight-sync from the FSDP policy worker. Eval-only has no policy worker, so
# wake_up() leaves the engine with corrupted weights -> pure gibberish output
# -> every turn parse-errors -> all scores 0. On the Qwen3.5 (qwen3_5) arch a
# sleep(level=2)+wake_up round-trip provably corrupts weights; level=1 is fine.
# colocate_all=false skips the sleep entirely so the engine keeps real weights.
exec "$PY" -m echo_rl.web_agent.eval_entrypoint --config "$CONFIG" \
  generator.eval_n_samples_per_prompt=1 \
  trainer.logger=console \
  trainer.placement.colocate_all=false \
  "${MODEL_PATH_OVERRIDE[@]}" \
  "$@"
