#!/usr/bin/env bash
# 9B vLLM-Triton smoke test on bonete61.
#
# Load Qwen3.5-9B in a vLLM engine and generate 1 token. Qwen3.5-9B's Gated
# DeltaNet layer JIT-compiles a Triton kernel on first forward; the goal of
# this script is to surface that compile (success or failure) without sinking
# GPU time into a full training run. ~5-10 min per run on 1 B200.
#
# Compare a known-bad image with a candidate base by submitting twice:
#   bash docker/submit_smoke_9b_triton.sh                      # default echo-rl:latest
#   IMAGE=aifrontiers.azurecr.io/<base>:<tag> \
#     JOB_TAG=smoke9bnvbase bash docker/submit_smoke_9b_triton.sh
#
# JOB_TAG controls the job-name segment so you can tell the two pods apart.
# (The workstream label is fixed to 'cua' for all jobs.)

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
JOB_TAG="${JOB_TAG:-smoke9b}"
# submit_job.sh requires --upload; we don't actually need any code, so point at
# a tiny stub directory (just the mini-web-agent dir itself is fine — we don't
# pip-install anything from it).
UPLOAD_DIR="${UPLOAD_DIR:-/data/t-yifeili/mini-web-agent/docker}"

export PATH="$HOME/.krew/bin:$PATH"
export PRIORITY="${PRIORITY:-medium}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-medium}"
# Submit under the 'cua' workstream by default; route JOB_TAG into JOB_NAME so
# the two comparison pods still get distinct names on the dashboard.
export PROJECT_NAME="${PROJECT_NAME:-cua}"
export JOB_NAME="${JOB_NAME:-${USER%@*}-${PRIORITY}-${JOB_TAG}-job}"

# FOLLOW_LOGS=1 streams the master pod logs until the pod exits (recommended:
# these pods abort-and-GC fast on failure, so streaming is the only reliable
# way to capture the traceback). Default on; set FOLLOW_LOGS=0 for fire-and-forget.
FOLLOW_ARGS=()
[[ "${FOLLOW_LOGS:-1}" == "1" ]] && FOLLOW_ARGS+=(--follow-logs)

bash "$SUBMIT" \
    --upload "$UPLOAD_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node 1 \
    --cpu 16 --memory 128Gi --shm 16Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    "${FOLLOW_ARGS[@]}" \
    --cmd 'set -e
echo "[smoke] hello from $JOB_NAME on $(hostname)"
source /run/secrets/echo-rl-creds/cred.sh
echo "[smoke] HF_HOME=$HF_HOME"

# Optional dep override: when running on a base image (e.g. the colleague-built
# nvidia25.08 one), transformers may be too old to recognize Qwen3.5 (`qwen3_5`
# model type). Allow upgrading transformers + installing missing deps without
# rebuilding the image.
if [[ -n "${EXTRA_PIP:-}" ]]; then
  echo "[smoke] === pip install ${EXTRA_PIP} ==="
  pip install --no-cache-dir ${EXTRA_PIP}
fi

# Baseline versions: probe.py runs as a real file so vllm worker subprocesses
# can re-import it via multiprocessing spawn (python -<<HEREDOC reads via
# `<stdin>` and vllm spawn workers try to re-exec the same path -> FileNotFound).
cat > /tmp/probe_versions.py <<PY
import torch, vllm
try:
    import triton; tv = triton.__version__
except Exception as e:
    tv = f"<no triton: {e}>"
try:
    import flashinfer; fi = flashinfer.__version__
except Exception as e:
    fi = f"<no flashinfer: {e}>"
try:
    import transformers; trv = transformers.__version__
except Exception as e:
    trv = f"<no transformers: {e}>"
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("triton      ", tv)
print("vllm        ", vllm.__version__)
print("flashinfer  ", fi)
print("transformers", trv)
PY
python /tmp/probe_versions.py

# Trigger the failing path: spin up vLLM with Qwen3.5-9B (GDN JIT compiles a
# Triton kernel on first forward). The `if __name__ == "__main__"` guard is
# REQUIRED: vllm V1 uses spawn for its EngineCore workers, which re-imports
# this file; without the guard the re-import re-runs LLM(...) and recursively
# spawns (RuntimeError: attempt to start a new process before bootstrapping).
cat > /tmp/probe_vllm_9b.py <<PY
from vllm import LLM, SamplingParams

def main():
    print("[smoke] constructing LLM Qwen/Qwen3.5-9B ...", flush=True)
    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    print("[smoke] generating ...", flush=True)
    out = llm.generate(["Hello, world!"], SamplingParams(max_tokens=8, temperature=0.0))
    print("[smoke] GENERATION OK:", repr(out[0].outputs[0].text), flush=True)

if __name__ == "__main__":
    main()
PY
python /tmp/probe_vllm_9b.py

echo "[smoke] DONE rc=0"
'
