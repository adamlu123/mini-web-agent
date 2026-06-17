#!/usr/bin/env bash
# ONLINE om2w eval of the 9B web-agent SFT checkpoint, with the rollout prompt /
# action-format / harness ALIGNED to the SFT training data (prompt_mode=sft,
# parser=bash, env sft_mode=true). This is the prompt-aligned counterpart to
# scripts/sft_eval_online_m2w_q35_9b.sh (which used the qwen35 tool-call prompt
# and is therefore mismatched to the <think>/<bash>/<answer> SFT model).
#
# What "aligned" means here (see configs/qwen35_9b_web_agent_easy_eval_sft.yaml):
#   - system + first-user <instructions> are the verbatim SFT prompt
#   - model emits <think>/<bash>/<answer>; parsed by BashAnswerParser
#   - env writes /workspace/task.json + a browser_session.py shim, the agent
#     self-launches a FRESH Browserbase session per command, and the OSW judge
#     scores the MODEL-authored screenshots (final_runs/run_*/screenshots/*.png)
#   - judge (osw_judge / WebJudge_Online_Mind2Web) is UNCHANGED -> same eval target
#
# Usage:
#   bash scripts/sft_eval_online_m2w_sft_aligned.sh                 # full easy train+val
#   SMOKE=1 bash scripts/sft_eval_online_m2w_sft_aligned.sh         # 1 task, concurrency 1
#   WEB_SFT_CKPT=/path/to/hf/ckpt RUN_TAG=sftA bash scripts/sft_eval_online_m2w_sft_aligned.sh
#
# Requires the same online prerequisites as run_local_eval.sh: 4 GPUs +
# Browserbase / judge creds (sourced below from cred.sh).
set -euo pipefail

REPO=/data/t-yifeili/mini-web-agent
PY=/data/t-yifeili/miniconda3/envs/echo-rl/bin/python
CONFIG="$REPO/configs/qwen35_9b_web_agent_easy_eval_sft.yaml"

export WEB_SFT_CKPT="${WEB_SFT_CKPT:-/data/t-yifeili/ckpts/websft_32k}"
[[ -d "$WEB_SFT_CKPT" ]] || { echo "[error] WEB_SFT_CKPT dir not found: $WEB_SFT_CKPT"; exit 1; }
ls "$WEB_SFT_CKPT"/*.safetensors >/dev/null 2>&1 || ls "$WEB_SFT_CKPT"/model.safetensors.index.json >/dev/null 2>&1 \
  || { echo "[error] $WEB_SFT_CKPT has no *.safetensors (not an HF-format weights dir)"; exit 1; }
[[ -f "$WEB_SFT_CKPT/chat_template.jinja" ]] || { echo "[error] $WEB_SFT_CKPT missing chat_template.jinja"; exit 1; }

# --- creds: browserbase project id + HF token (same as run_local_eval.sh) ---
source /home/luyadong/cred.sh
unset OPENAI_GATEWAY_API_KEY || true
export OPENAI_GATEWAY_ENDPOINT=''

export MINI_WEB_AGENT_ROOT=$REPO
export ECHO_RL_DATA=$REPO/data/web_agent
export OUTPUT_DIR=$REPO/eval_outputs/9b_easy_sftaligned_${RUN_TAG:+${RUN_TAG}_}$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Smoke knobs: 1 task, single concurrency (cheap, fast prompt-alignment check).
SMOKE_OVERRIDES=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_OVERRIDES=(
    generator.dataset_max_rows=1
    generator.agent_max_concurrency=1
    generator.max_concurrent_builds=1
  )
  echo "[run] SMOKE mode: 1 task, concurrency 1"
fi

echo "[run] SFT-ALIGNED online om2w eval"
echo "[run] CKPT=$WEB_SFT_CKPT"
echo "[run] CONFIG=$CONFIG"
echo "[run] OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi -L || true

cd "$REPO"
# colocate_all=false is REQUIRED for eval-only (see run_local_eval.sh comment).
exec "$PY" -m echo_rl.web_agent.eval_entrypoint --config "$CONFIG" \
  generator.eval_n_samples_per_prompt=1 \
  trainer.logger=console \
  trainer.placement.colocate_all=false \
  "${SMOKE_OVERRIDES[@]}" \
  "$@"
