#!/bin/bash
# Serve a Qwen3.5 checkpoint with vLLM for the SPB om2w eval (see
# config/eval/om2w_spb_vllm_*.yaml). Pair with scripts/run_om2w_vllm.sh.
#
#   CKPT=/path/to/ctx2-<variant>-hf-vlm GPUS=0,1 TP=2 bash scripts/serve_vllm_qwen35.sh
#
# Workarounds, both env-only so the shared phitrain venv is never modified:
#   FLASHINFER_DISABLE_VERSION_CHECK=1 -> flashinfer-cubin 0.6.13 vs flashinfer 0.6.14.
#   VLLM_USE_FLASHINFER_SAMPLER=0      -> flashinfer's sampling kernel fails to
#     JIT-compile against the installed CUB (BlockAdjacentDifference::FlagHeads
#     removed); fall back to vLLM's native sampler.
set -euo pipefail

REPO="${REPO:-/home/luyadong/sandbox/mini-web-agent}"
CKPT="${CKPT:-/data/yadonglu/hf/Qwen3.5-4B}"
SERVED_NAME="${SERVED_NAME:-sft_ckpt}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_BIN="${VLLM_BIN:-/data/yadonglu/venvs/phitrain/bin/vllm}"

# Qwen3.5 is hybrid (24 linear_attention + 8 full_attention layers) and vLLM keeps
# prefix caching opt-in for hybrids "while the feature matures"
# (engine/arg_utils.py:2518) -- supported, just off by default. Verified on
# Qwen3.5-4B: starts with enable_prefix_caching=True, reaches a 57.3% hit rate,
# and returns identical greedy output on a repeated prompt. Biggest win for the
# full24k config (append-only history = 100% reusable prefix); little effect for
# sw10, whose prefix shifts every turn once the window slides. Set
# PREFIX_CACHING=--no-enable-prefix-caching to opt back out.
PREFIX_CACHING="${PREFIX_CACHING:---enable-prefix-caching}"

# Train-aligned chat template (PIPELINE_SPB.md 2.2 step 3). WITHOUT this vLLM
# falls back to the checkpoint's stock Qwen3.5 template, which
#   1. STRIPS <think>...</think> from every history assistant turn -- defeating
#      the sft_state history replay in models/openrouter_model.py, and
#   2. prefills "<think>\n" into the generation prompt, so the model's output
#      starts after the opening tag.
# Both are train/eval mismatches for checkpoints SFT'd with the phitrain
# template. The branch also ships _prefill_think and _trim variants; pick the
# one matching how the checkpoint was tokenized, or set CHAT_TEMPLATE= (empty)
# to use whatever ships with the checkpoint.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$REPO/src/miniswewebagent/config/eval/qwen3_5_train_aligned.jinja}"
template_arg=()
[ -n "$CHAT_TEMPLATE" ] && template_arg=(--chat-template "$CHAT_TEMPLATE")

export CUDA_VISIBLE_DEVICES="$GPUS"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$VLLM_BIN" serve "$CKPT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  "${template_arg[@]}" \
  $PREFIX_CACHING
