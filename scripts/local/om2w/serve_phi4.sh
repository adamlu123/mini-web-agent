#!/usr/bin/env bash
# Serve the Phi-4 rlscaling intermediate checkpoint with vLLM for the om2w eval.
# Phi-4 analogue of serve_qwen35.sh; pair with scripts/local/om2w/shards.sh.
#
#   CKPT=/data/yadonglu/ckpts/phi4-rlscaling-resume-9b184-6000-llama \
#   GPUS=0,1 TP=2 PORT=8000 bash scripts/local/om2w/serve_phi4.sh
#
# Same two host workarounds as the Qwen script, env-only so the shared phitrain
# venv is never modified:
#   FLASHINFER_DISABLE_VERSION_CHECK=1 -> flashinfer-cubin 0.6.13 vs flashinfer 0.6.14.
#   VLLM_USE_FLASHINFER_SAMPLER=0      -> flashinfer's sampling kernel fails to
#     JIT-compile against the installed CUB (BlockAdjacentDifference::FlagHeads
#     removed); fall back to vLLM's native sampler.
#
# About this checkpoint (pulled off the lambda PVC with pull_ckpt_from_pvc.sh):
#   architectures=LlamaForCausalLM, model_type=llama -> vLLM's native Llama path.
#   The stray configuration_llama.py / modeling_llama.py in the directory are NOT
#   referenced by an `auto_map` in config.json, so nothing needs trust_remote_code.
#   hidden 5120 x 40 layers, vocab 100352, untied embeddings => ~14.7B params,
#   stored as float32 (58 GB on disk). "9b184" in the run name is a job hash,
#   not a parameter count.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CKPT="${CKPT:-/data/yadonglu/ckpts/phi4-rlscaling-resume-9b184-6000-llama}"
SERVED_NAME="${SERVED_NAME:-sft_ckpt}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1}"
# TP MUST DIVIDE num_key_value_heads=10, so 2 is the only workable size on a
# 4-GPU box: vLLM shards KV heads across ranks and needs
# total_num_kv_heads % tp == 0 (or tp % total_num_kv_heads == 0).
# TP=4 fails with "Total number of attention heads (10) must be divisible by
# tensor parallel size (4)". TP=1 also works but needs ~30 GB of weights on one
# 46 GB A6000, leaving little room for KV cache.
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # config.json max_position_embeddings
VLLM_BIN="${VLLM_BIN:-/data/yadonglu/venvs/phitrain/bin/vllm}"

# The checkpoint is stored float32 (config.json "dtype": "float32"). Serving it
# as-is would pin ~58 GB of weights and crowd out the KV cache; A6000 is Ampere
# and does bf16 natively, so cast on load: ~29 GB total, ~15 GB per rank at TP=2.
DTYPE="${DTYPE:---dtype bfloat16}"

# Unlike the Qwen3.5 case there is NO train-aligned template to force here: this
# is a PRETRAINING intermediate, not an SFT'd checkpoint, so there is no training
# chat format to align to. The directory ships its own chat_template.jinja (the
# Phi <|im_start|>role<|im_sep|>...<|im_end|> format, system/user/assistant only,
# no tool-calling section) and vLLM picks it up from the model directory. Set
# CHAT_TEMPLATE=/path/to.jinja to override.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
template_arg=()
[ -n "$CHAT_TEMPLATE" ] && template_arg=(--chat-template "$CHAT_TEMPLATE")

export CUDA_VISIBLE_DEVICES="$GPUS"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[serve_phi4] ckpt=$CKPT gpus=$GPUS tp=$TP port=$PORT len=$MAX_MODEL_LEN"
exec "$VLLM_BIN" serve "$CKPT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  $DTYPE \
  "${template_arg[@]}" \
  --enable-prefix-caching
