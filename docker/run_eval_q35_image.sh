#!/usr/bin/env bash
# In-pod CLUSTER EVAL driver for the web-agent on the *generic* qwen3.5 image
#   aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415
#
# Cluster-side equivalent of the LOCAL run_local_eval.sh: loads HF weights into
# vLLM and runs a single evaluate() pass over the om2w easy train+val parquets
# (each scored separately), logging eval/<data_source>/... to the console.
# No FSDP, no optimizer, no training -- just rollout + OSW judge scoring.
#
# Same RL/eval stack bootstrap as docker/run_train_q35_image.sh (resolve the few
# deps the image lacks WITHOUT clobbering its baked torch2.10/te2.13/vllm0.18
# stack, --no-deps the 4 editables, ray2.55 placement-group shim, playwright
# chromium, creds). The ONLY difference vs the train driver: it launches
# `python -m echo_rl.web_agent.eval_entrypoint` with the eval-only overrides
# (eval_n_samples_per_prompt=1, logger=console, colocate_all=false) that
# run_local_eval.sh uses, optionally pointing the policy weights at a trained
# HF checkpoint via EVAL_CKPT.
#
# Required env (auto-injected by submit_job.sh):
#   PVC_MOUNT, USER_ALIAS, JOB_NAME
# Optional env (forwarded via --extra-env-vars, all have defaults):
#   EVAL_CONFIG  -- config path relative to mini-web-agent/ (default: configs/qwen35_4b_web_agent_easy_eval.yaml)
#   EVAL_CKPT    -- HuggingFace-format weights dir to eval (default: empty -> base weights from the config)
#   EVAL_RUN_TAG -- short tag mixed into the eval OUTPUT_DIR (e.g. step1, merged)
# Secrets (mounted as volumes, same as the train driver):
#   /run/secrets/echo-rl-creds/cred.sh         -- BROWSERBASE_*, HF_*, gateway key
#   /run/secrets/echo-rl-openai/OPENAI_API_KEY -- working sk-proj judge key
#
# NOTE: colocate_all=false is REQUIRED for eval-only on the qwen3_5 arch -- with
# colocate_all=true SkyRL sleeps the vLLM engine at level=2 right after startup
# expecting a later NCCL weight-sync from an FSDP policy worker; eval-only has no
# such worker, so the engine wakes with corrupted weights -> all scores 0. See
# run_local_eval.sh for the full write-up.

set -e

echo "[eval] q35-image EVAL pod $JOB_NAME on $(hostname)"
EVAL_CONFIG="${EVAL_CONFIG:-configs/qwen35_4b_web_agent_easy_eval.yaml}"
echo "[eval] EVAL_CONFIG=$EVAL_CONFIG  EVAL_CKPT=${EVAL_CKPT:-<base weights from config>}  RUN_TAG=${EVAL_RUN_TAG:-}"

CODE_ROOT=$PVC_MOUNT/$USER_ALIAS/code
UPLOAD_ROOT=$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME
ENV_ROOT=$PVC_MOUNT/$USER_ALIAS/envs/q35-image
mkdir -p "$CODE_ROOT" "$ENV_ROOT"

echo '[eval] === copy uploaded code to stable PVC path (idempotent; no-op if already synced by a prior phase) ==='
if ! command -v rsync >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq rsync
fi
# Only sync what is present in the upload (the combined train+eval driver may
# have already rsynced these; re-running is a cheap no-op).
[ -d "$UPLOAD_ROOT/SkyRL" ]          && rsync -a --no-perms --no-owner --no-group --no-times "$UPLOAD_ROOT/SkyRL/"          "$CODE_ROOT/SkyRL/"
[ -d "$UPLOAD_ROOT/mini-web-agent" ] && rsync -a --no-perms --no-owner --no-group --no-times "$UPLOAD_ROOT/mini-web-agent/" "$CODE_ROOT/mini-web-agent/"

REQ="$CODE_ROOT/mini-web-agent/docker/requirements.txt"
MISSING="$ENV_ROOT/requirements.missing.txt"

# Skip the (slow) dep resolve+install if a prior phase in THIS pod already did
# it (marker file). The combined train+eval driver sets EVAL_SKIP_BOOTSTRAP=1
# after the RL stack is up to avoid re-installing.
if [ "${EVAL_SKIP_BOOTSTRAP:-0}" != "1" ]; then
  echo '[eval] === resolve missing deps (keep everything the image already bakes) ==='
  echo "[eval] python -> $(command -v python) ; $(python -V 2>&1)"
  python - "$REQ" "$MISSING" <<'PY'
import sys, re
from importlib.metadata import distributions

def canon(n):
    return re.sub(r'[-_.]+', '-', n).strip().lower()

# EXCLUDE: the compiled CUDA / torch / kernel stack. The NGC base image provides
# these (many as SYSTEM libs invisible to pip metadata), so installing the
# requirements.txt pins would CLOBBER the image's working torch2.10/vllm0.18
# stack and break torch/vllm. Never reinstall these.
EXCLUDE_PREFIX = ('nvidia-', 'cuda-', 'nixl')
EXCLUDE_EXACT = {
    'torch', 'torchaudio', 'torchvision', 'torchdata', 'torch-c-dlpack-ext',
    'triton', 'vllm',
    'flash-attn', 'flash-linear-attention', 'fla-core', 'causal-conv1d',
    'flashinfer-cubin', 'flashinfer-python',
    'apache-tvm-ffi', 'tilelang', 'quack-kernels', 'xgrammar',
}

def is_image_stack(name):
    c = canon(name)
    return c in EXCLUDE_EXACT or c.startswith(EXCLUDE_PREFIX)

# FORCE_UPGRADE: app deps the image ships too OLD/mismatched for echo_rl.
#   omegaconf -> image 2.0.0 lacks the oc.env resolver (need >=2.1; req 2.3.0)
#   antlr4-python3-runtime -> omegaconf 2.3.0 is version-locked to 4.9.3
#   ray -> SkyRL pins 2.51.1; image ray has drifted (2.55 moved a symbol)
FORCE_UPGRADE = {'omegaconf', 'antlr4-python3-runtime', 'ray'}

req_path, out_path = sys.argv[1], sys.argv[2]
installed = {canon(d.metadata['Name']) for d in distributions()
             if d.metadata.get('Name')}

kept, skipped, excluded, forced = [], [], [], []
with open(req_path) as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        name = re.split(r'[=<>!~ ]', s, maxsplit=1)[0]
        if canon(name) in FORCE_UPGRADE:
            kept.append(s); forced.append(s)
        elif is_image_stack(name):
            excluded.append(s)
        elif canon(name) in installed:
            skipped.append(s)
        else:
            kept.append(s)

with open(out_path, 'w') as f:
    f.write('\n'.join(kept) + ('\n' if kept else ''))

print(f"[eval] excluded {len(excluded)} compiled-stack reqs (use image's CUDA/torch/vllm)")
print(f"[eval] image already provides {len(skipped)} other reqs (kept as-is)")
print(f"[eval] installing {len(kept)} reqs --no-deps (force-upgraded too-old: {forced})")
PY

  if [ -s "$MISSING" ]; then
    pip install --no-deps -r "$MISSING"
  else
    echo '[eval] nothing missing -- image already satisfies requirements.txt'
  fi

  echo '[eval] === pip install editables into the IMAGE Python (no venv, no uv) ==='
  pip install --no-deps --no-build-isolation \
      -e "$CODE_ROOT/SkyRL" \
      -e "$CODE_ROOT/SkyRL/skyrl-gym" \
      -e "$CODE_ROOT/SkyRL/skyrl-agent" \
      -e "$CODE_ROOT/mini-web-agent"
  python -c "import wandb" 2>/dev/null || pip install wandb

  echo '[eval] === ensure ray.util.placement_group re-exports PlacementGroupSchedulingStrategy ==='
  if python -c "from ray.util.placement_group import PlacementGroupSchedulingStrategy" 2>/dev/null; then
    echo '[eval] ray import already OK (no shim needed)'
  else
    PG_FILE=$(python - <<'PY'
import importlib.util
s = importlib.util.find_spec('ray.util.placement_group')
print(s.origin if s and s.origin else '')
PY
)
    if [ -n "$PG_FILE" ] && [ -f "$PG_FILE" ]; then
      echo 'from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # compat: ray2.55 moved it' >> "$PG_FILE"
      echo "[eval] patched $PG_FILE"
    fi
  fi

  echo '[eval] === install playwright chromium (not baked in this image) ==='
  playwright install --with-deps chromium || playwright install chromium || \
      echo '[eval] WARN: playwright browser install failed (ok if using browserbase)'
fi

echo '[eval] === source creds + route OSW judge to api.openai.com ==='
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY || true
# phyagi gateway key in cred.sh is dead; use the working sk-proj key from the
# echo-rl-openai secret and route the OSW judge straight to api.openai.com
# (empty gateway endpoint -> direct).
if [ -f /run/secrets/echo-rl-openai/OPENAI_API_KEY ]; then
  export OPENAI_API_KEY="$(cat /run/secrets/echo-rl-openai/OPENAI_API_KEY)"
  export OPENAI_GATEWAY_ENDPOINT=''
  echo '[eval] OPENAI_API_KEY set from echo-rl-openai secret; judge -> api.openai.com'
fi

echo '[eval] === env paths ==='
export MINI_WEB_AGENT_ROOT=$CODE_ROOT/mini-web-agent
export ECHO_RL_DATA=$CODE_ROOT/mini-web-agent/data/web_agent
export OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME/eval${EVAL_RUN_TAG:+_${EVAL_RUN_TAG}}
mkdir -p "$OUTPUT_DIR"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_DEBUG="${NCCL_DEBUG_OVERRIDE:-WARN}"
echo "[eval] MINI_WEB_AGENT_ROOT=$MINI_WEB_AGENT_ROOT"
echo "[eval] ECHO_RL_DATA=$ECHO_RL_DATA"
echo "[eval] OUTPUT_DIR=$OUTPUT_DIR"

# Optional policy-weights override (a trained, HF-format ckpt). When empty the
# eval runs the base weights from the config (matches run_local_eval.sh sans CKPT).
CKPT_OVERRIDE=()
if [ -n "${EVAL_CKPT:-}" ]; then
  if [ -d "$EVAL_CKPT" ]; then
    CKPT_OVERRIDE=( "trainer.policy.model.path=${EVAL_CKPT}" )
    echo "[eval] policy weights -> trained ckpt: $EVAL_CKPT"
    # The SFT-ALIGNED eval config (configs/*_eval_sft.yaml) resolves
    # tokenizer_path / chat_template_path / model path from ${oc.env:WEB_SFT_CKPT}
    # so the rollout prompt + action format + chat template match SFT training.
    # Point it at the same ckpt (no-op for the non-sft config, which ignores it).
    export WEB_SFT_CKPT="${WEB_SFT_CKPT:-$EVAL_CKPT}"
    echo "[eval] WEB_SFT_CKPT=$WEB_SFT_CKPT (used by the *_eval_sft.yaml config)"
  else
    echo "[eval][warn] EVAL_CKPT='$EVAL_CKPT' is not a dir; falling back to base weights from config"
  fi
fi

echo '[eval] === pre-flight ==='
nvidia-smi -L
python - <<'PY'
import torch, vllm, omegaconf
import echo_rl, skyrl, skyrl_gym, skyrl_agent  # editables
import vllm_router  # noqa: F401
from omegaconf import OmegaConf
assert OmegaConf.create({"x": "${oc.env:HOME,/tmp}"}).x, "oc.env resolver missing"
from skyrl.backends.skyrl_train.inference_engines.utils import PlacementGroupSchedulingStrategy  # noqa
print("torch", torch.__version__, "| vllm", vllm.__version__, "| omegaconf", omegaconf.__version__)
print("[eval] pre-flight OK")
PY

# vLLM engine setup for EVAL-ONLY (colocate_all=false). Two cluster-specific
# pitfalls the training-tuned config hits here:
#   1. num_engines=4 + colocate_all=false PILES all 4 vLLM engines onto GPU 0
#      (Ray doesn't spread them on this eval path) -> CUDA OOM on GPU 0 while
#      GPUs 1-3 sit idle. Fix: run a SINGLE engine (num_engines=1, tp=1); a 9B
#      model fits with room to spare on one 180GB B200, and eval only scores ~80
#      tasks so one engine is plenty.
#   2. max_num_batched_tokens=262144 + enforce_eager=false (CUDA-graph capture)
#      bloat memory and slow startup. Skip graphs and shrink the prefill batch
#      (chunked prefill stays on).
# All overridable via EVAL_ENGINE_OVERRIDES (space-separated dotlist; empty="").
EVAL_ENGINE_OVERRIDES="${EVAL_ENGINE_OVERRIDES-generator.inference_engine.num_engines=1 generator.inference_engine.tensor_parallel_size=1 generator.inference_engine.enforce_eager=true generator.inference_engine.gpu_memory_utilization=0.85 generator.inference_engine.max_num_batched_tokens=32768}"
# shellcheck disable=SC2206
ENGINE_OVERRIDES=( $EVAL_ENGINE_OVERRIDES )

echo "[eval] === launching eval: $EVAL_CONFIG ==="
echo "[eval] engine overrides: ${ENGINE_OVERRIDES[*]:-<none>}"
# cwd = mini-web-agent root so the config's relative chat_template_path and the
# `configs/...` config path resolve (matches run_local_eval.sh's `cd $REPO`).
cd "$CODE_ROOT/mini-web-agent"
RC=0
RAY_DEDUP_LOGS=0 python -m echo_rl.web_agent.eval_entrypoint --config "$EVAL_CONFIG" \
    generator.eval_n_samples_per_prompt=1 \
    trainer.logger=console \
    trainer.placement.colocate_all=false \
    "${ENGINE_OVERRIDES[@]}" \
    "${CKPT_OVERRIDE[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/eval_console.log"
RC=${PIPESTATUS[0]}
echo "[eval] eval exited rc=$RC"

# === on failure, surface the REAL root cause =================================
# vLLM v1 spawns EngineCore in subprocesses whose tracebacks do NOT reach this
# stdout (the parent only prints "Engine core initialization failed. Failed core
# proc(s): {}"). The actual exception lands in the SkyRL infra log and the ray
# worker logs. Dump their tails so a failed eval is diagnosable from pod stdout
# alone (no PVC access needed).
if [ "$RC" -ne 0 ]; then
  echo "[eval][diag] ===== eval failed (rc=$RC); dumping root-cause logs ====="
  echo "[eval][diag] --- SkyRL infra log(s) under $OUTPUT_DIR/skyrl_logs ---"
  for f in "$OUTPUT_DIR"/skyrl_logs/infra-*.log; do
    [ -f "$f" ] && { echo "[eval][diag] >>> $f (tail)"; tail -n 80 "$f"; }
  done
  echo "[eval][diag] --- ray worker stderr with tracebacks (most recent) ---"
  for d in /tmp/ray/session_latest/logs /root/ray/session_latest/logs; do
    [ -d "$d" ] || continue
    # worker-*.err / *VLLMServerActor* often hold the EngineCore traceback
    grep -lE "Traceback|Error|EngineCore|CUDA|out of memory|RuntimeError" "$d"/*.err "$d"/*.out 2>/dev/null \
      | head -4 | while read -r lf; do echo "[eval][diag] >>> $lf (tail)"; tail -n 60 "$lf"; done
  done
  echo "[eval][diag] ===== end root-cause dump ====="
fi
exit "$RC"
