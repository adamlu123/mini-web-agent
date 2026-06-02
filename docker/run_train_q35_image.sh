#!/usr/bin/env bash
# In-pod driver for a PURE-IMAGE TRAINING run on the *generic* qwen3.5 image
#   aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415
# rather than our own t-yifeili/echo-rl image.
#
# Same bootstrap as run_debug_q35_image.sh (deny-list deps, force-upgrade
# omegaconf/antlr4, install vllm-router, --no-deps -e the 4 editables, ray
# placement_group shim). The ONLY difference vs the debug driver: instead of
# writing an activate script and `sleep infinity`, this one sources creds +
# exports env, then LAUNCHES training (python -m echo_rl.web_agent.entrypoint
# --config $TRAIN_CONFIG) and exits with the trainer's rc.
#
# Required env (forwarded by submit_train_q35_image.sh via --extra-env-vars):
#   TRAIN_CONFIG   -- config path relative to SkyRL/ (e.g. echo_configs/...yaml)
#
# Why a separate driver from run_debug_pure_image.sh:
#   run_debug_pure_image.sh assumes the echo-rl image (which BAKES the full
#   269-pkg docker/requirements.txt freeze), so it only `pip install --no-deps
#   -e`'s the four editables. This image does NOT bake that freeze -- it only
#   ships the heavy runtime (torch 2.10, transformer-engine 2.13, deepspeed,
#   flash-attn, vllm 0.18.0) + the CUDA wheels. So on a plain `--no-deps -e`
#   the very first `import playwright`/skyrl dep would fail.
#
# Strategy ("keep image stack, add rest"): the whole point of this image is
# that qwen3.5 works on its newer vllm 0.18.0 / torch 2.10 / TE stack (vs. the
# gibberish corruption seen on the echo-rl image). So we MUST NOT clobber that
# stack. We therefore skip every requirement whose package is ALREADY present
# in the image (any version) and `--no-deps` install only the genuinely-missing
# ones. That keeps vllm 0.18.0 and its entire baked dependency closure intact
# and just layers on the pure-add deps (playwright, browserbase, datasets,
# litellm, ...). Then `--no-deps -e` the four editables, exactly like the
# pure-image driver.
#
# UPLOADED with the repo and executed via a tiny `--cmd` one-liner (see
# docker/submit_debug_q35_image.sh) -- keeps the `kubectl create` body clear of
# the Cloudflare WAF that blocks big inline shell preambles.
#
# Required env (auto-injected by submit_job.sh):
#   PVC_MOUNT, USER_ALIAS, JOB_NAME
# Secrets (mounted as volumes):
#   /run/secrets/echo-rl-creds/cred.sh         -- BROWSERBASE_*, HF_*, gateway key
#   /run/secrets/echo-rl-openai/OPENAI_API_KEY -- working sk-proj judge key

set -e

# Unlike the debug driver, a TRAINING pod SHOULD exit on bootstrap failure
# (don't hold 8 B200s sleeping) -- so NO sleep-infinity ERR trap here; set -e
# propagates and Volcano marks the job failed.
echo "[boot] q35-image TRAIN pod $JOB_NAME on $(hostname)"
echo "[boot] TRAIN_CONFIG=${TRAIN_CONFIG:?TRAIN_CONFIG not set}"

CODE_ROOT=$PVC_MOUNT/$USER_ALIAS/code
UPLOAD_ROOT=$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME
OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
ENV_ROOT=$PVC_MOUNT/$USER_ALIAS/envs/q35-image
mkdir -p "$CODE_ROOT" "$OUTPUT_DIR" "$ENV_ROOT"

echo '[boot] === copy uploaded code to stable PVC path ==='
if ! command -v rsync >/dev/null 2>&1; then
  echo '[boot] installing rsync (one-time per pod)...'
  apt-get update -qq && apt-get install -y -qq rsync
fi
rsync -a --delete --no-perms --no-owner --no-group --no-times \
    "$UPLOAD_ROOT/SkyRL/"          "$CODE_ROOT/SkyRL/"
rsync -a --delete --no-perms --no-owner --no-group --no-times \
    "$UPLOAD_ROOT/mini-web-agent/" "$CODE_ROOT/mini-web-agent/"

REQ="$CODE_ROOT/mini-web-agent/docker/requirements.txt"
MISSING="$ENV_ROOT/requirements.missing.txt"

echo '[boot] === resolve missing deps (keep everything the image already bakes) ==='
echo "[boot] python -> $(command -v python) ; $(python -V 2>&1)"
python - "$REQ" "$MISSING" <<'PY'
import sys, re
from importlib.metadata import distributions

def canon(n):
    return re.sub(r'[-_.]+', '-', n).strip().lower()

# EXCLUDE: the compiled CUDA / torch / kernel stack. The NGC base image provides
# these -- crucially, many (NCCL, cuBLAS, cuDNN, ...) as SYSTEM libraries that
# are INVISIBLE to pip metadata. So the skip-if-already-installed check below
# does NOT catch them, and installing the requirements.txt pins CLOBBERS the
# image's stack: e.g. nvidia-nccl-cu12==2.27.5 shadows the image's newer NCCL
# and breaks torch with `undefined symbol: ncclAlltoAll` -> torch/vllm dead.
# Never reinstall these; always use whatever the image baked. (This is THE
# reason to use this image -- its torch2.10/te2.13/vllm0.18 stack runs qwen3.5.)
EXCLUDE_PREFIX = ('nvidia-', 'cuda-', 'nixl')
# NOTE: vllm-router is deliberately NOT here -- despite the name it is a pure
# Python routing proxy (not part of the cu130 compiled stack), and the image
# does NOT bake it, so it must be pip-installed or SkyRL's inference-server
# setup dies with `ModuleNotFoundError: No module named 'vllm_router'`.
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

# FORCE_UPGRADE: app deps the image ships at a version too OLD/mismatched for
# echo_rl, so we (re)install the requirements.txt pin even though they're already
# present. This image is otherwise a near-superset of requirements.txt, EXCEPT:
#   omegaconf -> image has 2.0.0; echo_configs use `${oc.env:...}`, a resolver
#     added in omegaconf 2.1, so 2.0.0 fails at config load. req pin is 2.3.0.
#   antlr4-python3-runtime -> omegaconf 2.3.0's ANTLR-generated grammar is
#     version-LOCKED to 4.9.3; the image ships 4.11.0, which makes omegaconf
#     fail to import (`Could not deserialize ATN with version`). Only omegaconf
#     uses antlr4 here, so pinning it back to 4.9.3 is safe.
# (NOT for compiled-stack pkgs -- those stay on the image's cu130 versions.)
#   ray -> SkyRL pins ray==2.51.1 and is the only ray user (vLLM uses the `mp`
#     executor, not ray); the image's ray has DRIFTED across rebuilds (2.51->2.55)
#     and 2.55 removed PlacementGroupSchedulingStrategy from ray.util.placement_group.
#     Pinning ray to SkyRL's 2.51.1 matches SkyRL + is immune to image ray drift.
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
            kept.append(s)               # too-old in image -> install req pin
            forced.append(s)
        elif is_image_stack(name):
            excluded.append(s)           # image owns it (maybe via system libs)
        elif canon(name) in installed:
            skipped.append(s)            # already a pip dist in the image
        else:
            kept.append(s)               # genuine pure-add app dep -> install

with open(out_path, 'w') as f:
    f.write('\n'.join(kept) + ('\n' if kept else ''))

print(f"[boot] excluded {len(excluded)} compiled-stack reqs (use image's CUDA/torch/vllm)")
print(f"[boot] image already provides {len(skipped)} other reqs (kept as-is)")
print(f"[boot] installing {len(kept)} reqs --no-deps (force-upgraded too-old: {forced})")
PY

if [ -s "$MISSING" ]; then
  pip install --no-deps -r "$MISSING"
else
  echo '[boot] nothing missing -- image already satisfies requirements.txt'
fi

echo '[boot] === pip install editables into the IMAGE Python (no venv, no uv) ==='
pip install --no-deps --no-build-isolation \
    -e "$CODE_ROOT/SkyRL" \
    -e "$CODE_ROOT/SkyRL/skyrl-gym" \
    -e "$CODE_ROOT/SkyRL/skyrl-agent" \
    -e "$CODE_ROOT/mini-web-agent"
python -c "import wandb" 2>/dev/null || pip install wandb

# ------------------------------------------------------------------
# Defensive ray shim: ray 2.55 (this image) REMOVED
# PlacementGroupSchedulingStrategy from the `ray.util.placement_group` submodule
# file, but SkyRL imports it from there in several spots. The SkyRL source is
# already patched to use the canonical `ray.util.scheduling_strategies`, but if
# a STALE (unpatched) SkyRL ever gets uploaded this would crash the run. Re-export
# the symbol into the container's ray submodule so EVERY import site works
# regardless of SkyRL state. Idempotent: only appends if the import is broken.
# ------------------------------------------------------------------
echo '[boot] === ensure ray.util.placement_group re-exports PlacementGroupSchedulingStrategy ==='
if python -c "from ray.util.placement_group import PlacementGroupSchedulingStrategy" 2>/dev/null; then
  echo '[boot] ray import already OK (no shim needed)'
else
  PG_FILE=$(python - <<'PY'
import importlib.util
s = importlib.util.find_spec('ray.util.placement_group')
print(s.origin if s and s.origin else '')
PY
)
  if [ -n "$PG_FILE" ] && [ -f "$PG_FILE" ]; then
    echo 'from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # compat: ray2.55 moved it' >> "$PG_FILE"
    echo "[boot] patched $PG_FILE"
  fi
fi

echo '[boot] === install playwright chromium (not baked in this image) ==='
# Web-agent rollouts can drive a local Chromium; the generic image ships no
# browsers. Non-fatal -- browserbase-backed runs do not need a local browser.
playwright install --with-deps chromium || playwright install chromium || \
    echo '[boot] WARN: playwright browser install failed (ok if using browserbase)'

echo '[boot] === source creds + route OSW judge to api.openai.com ==='
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY
# The phyagi gateway key in cred.sh is dead; use the working sk-proj key from the
# echo-rl-openai secret and route the OSW judge straight to api.openai.com.
if [ -f /run/secrets/echo-rl-openai/OPENAI_API_KEY ]; then
  export OPENAI_API_KEY="$(cat /run/secrets/echo-rl-openai/OPENAI_API_KEY)"
  export OPENAI_GATEWAY_ENDPOINT=''
  echo '[boot] OPENAI_API_KEY set from echo-rl-openai secret; judge -> api.openai.com'
fi

echo '[boot] === env paths ==='
export MINI_WEB_AGENT_ROOT=$CODE_ROOT/mini-web-agent
export ECHO_RL_DATA=$CODE_ROOT/mini-web-agent/data/web_agent
export OUTPUT_DIR=$OUTPUT_DIR
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
echo "[boot] MINI_WEB_AGENT_ROOT=$MINI_WEB_AGENT_ROOT"
echo "[boot] ECHO_RL_DATA=$ECHO_RL_DATA"
echo "[boot] OUTPUT_DIR=$OUTPUT_DIR"

echo '[boot] === pre-flight (validates every fix this driver applies) ==='
nvidia-smi -L
python - <<'PY'
import torch, vllm, transformers, wandb, omegaconf
# editables (importing skyrl_agent walks the chain that hits the ray import)
import echo_rl, skyrl, skyrl_gym, skyrl_agent
# vllm-router: missing on this image (was wrongly deny-listed); SkyRL needs it
import vllm_router  # noqa: F401
# omegaconf oc.env resolver: needs >=2.1 (image shipped 2.0.0 -> force-upgraded)
from omegaconf import OmegaConf
assert OmegaConf.create({"x": "${oc.env:HOME,/tmp}"}).x, "oc.env resolver missing"
# skyrl ray import path (ray2.55 moved PlacementGroupSchedulingStrategy)
from skyrl.backends.skyrl_train.inference_engines.utils import PlacementGroupSchedulingStrategy  # noqa
print("torch", torch.__version__, "| vllm", vllm.__version__,
      "| omegaconf", omegaconf.__version__)
print("imports + vllm_router + omegaconf oc.env + skyrl ray-import: ALL OK")
PY

echo "[boot] === launching training: $TRAIN_CONFIG (no time cap) ==="
cd "$CODE_ROOT/SkyRL"
RAY_DEDUP_LOGS=0 python -m echo_rl.web_agent.entrypoint --config "$TRAIN_CONFIG" \
    2>&1 | tee -a "$OUTPUT_DIR/console.log"
RC=${PIPESTATUS[0]}
echo "[boot] training exited rc=$RC"
echo '[boot] === done ==='
exit "$RC"
