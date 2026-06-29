#!/usr/bin/env bash
# 在 qwen3.5 通用镜像里运行历史 miniswewebagent OM2W harness：
# 1. 使用 pod/PVC 上的 HF 格式 SFT checkpoint 启动本地 vLLM；
# 2. 用今天仓库里的 miniswewebagent harness 跑全量 Online-Mind2Web；
# 3. 输出 generation、逐任务 result.json、judge 汇总到 PVC outputs 目录。

set -euo pipefail

: "${PVC_MOUNT:?PVC_MOUNT not set}"
: "${USER_ALIAS:?USER_ALIAS not set}"
: "${JOB_NAME:?JOB_NAME not set}"

UPLOAD_ROOT="$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME"
REPO="${REPO:-$UPLOAD_ROOT/mini-web-agent}"
ENV_ROOT="$PVC_MOUNT/$USER_ALIAS/envs/q35-mini-harness"
REQ="$REPO/docker/requirements.txt"
MISSING="$ENV_ROOT/requirements.missing.txt"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"

EVAL_CKPT="${EVAL_CKPT:-$PVC_MOUNT/$USER_ALIAS/models/qwen35_9b/full/web_agent_state_debug_latest_0623}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-state_debug_0623_fixed_harness}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/om2w_sft_state_debug_vllm_sft_ckpt.yaml}"
TASK_LEVEL="${TASK_LEVEL:-all}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-8}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"

TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
OUTPUT_DIR="${OUTPUT_DIR:-$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME/mini_harness_${EVAL_RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_DIR/logs}"

echo "[mini-harness] job=$JOB_NAME host=$(hostname)"
echo "[mini-harness] repo=$REPO"
echo "[mini-harness] ckpt=$EVAL_CKPT"
echo "[mini-harness] benchmark=$BENCHMARK_CONFIG level=$TASK_LEVEL limit=$LIMIT workers=$WORKERS"
echo "[mini-harness] output=$OUTPUT_DIR logs=$LOG_ROOT"

[[ -d "$REPO" ]] || { echo "[mini-harness][error] repo not found: $REPO"; exit 1; }
[[ -f "$REQ" ]] || { echo "[mini-harness][error] requirements not found: $REQ"; exit 1; }
[[ -f "$CREDS_FILE" ]] || { echo "[mini-harness][error] credentials file not found: $CREDS_FILE"; exit 1; }
[[ -d "$EVAL_CKPT" ]] || { echo "[mini-harness][error] ckpt dir not found: $EVAL_CKPT"; exit 1; }
if ! compgen -G "$EVAL_CKPT/*.safetensors" >/dev/null && [[ ! -f "$EVAL_CKPT/model.safetensors.index.json" ]]; then
  echo "[mini-harness][error] ckpt is not HF safetensors format: $EVAL_CKPT"
  exit 1
fi

mkdir -p "$ENV_ROOT" "$OUTPUT_DIR" "$LOG_ROOT"

echo '[mini-harness] === install missing python deps without touching image CUDA/torch/vLLM stack ==='
python - "$REQ" "$MISSING" <<'PY'
import re
import sys
from importlib.metadata import distributions


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


exclude_prefix = ("nvidia-", "cuda-", "nixl")
exclude_exact = {
    "torch", "torchaudio", "torchvision", "torchdata", "torch-c-dlpack-ext",
    "triton", "vllm", "flash-attn", "flash-linear-attention", "fla-core",
    "causal-conv1d", "flashinfer-cubin", "flashinfer-python", "apache-tvm-ffi",
    "tilelang", "quack-kernels", "xgrammar",
}
force_upgrade = {"omegaconf", "antlr4-python3-runtime", "ray"}


def image_stack(name: str) -> bool:
    normalized = canon(name)
    return normalized in exclude_exact or normalized.startswith(exclude_prefix)


req_path, out_path = sys.argv[1], sys.argv[2]
installed = {canon(dist.metadata["Name"]) for dist in distributions() if dist.metadata.get("Name")}
kept, skipped, excluded, forced = [], [], [], []

with open(req_path, encoding="utf-8") as handle:
    for line in handle:
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name = re.split(r"[=<>!~ ]", requirement, maxsplit=1)[0]
        normalized = canon(name)
        if normalized in force_upgrade:
            kept.append(requirement)
            forced.append(requirement)
        elif image_stack(name):
            excluded.append(requirement)
        elif normalized in installed:
            skipped.append(requirement)
        else:
            kept.append(requirement)

with open(out_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(kept) + ("\n" if kept else ""))

print(f"[mini-harness] excluded compiled-stack reqs: {len(excluded)}")
print(f"[mini-harness] already installed reqs: {len(skipped)}")
print(f"[mini-harness] installing reqs: {len(kept)} forced={forced}")
PY

if [[ -s "$MISSING" ]]; then
  pip install --no-deps -r "$MISSING"
else
  echo '[mini-harness] no missing deps'
fi

pip install --no-deps --no-build-isolation -e "$REPO"
python -c "import openai, rich, typer, browserbase, miniswewebagent; print('[mini-harness] import preflight OK')"

echo '[mini-harness] === source secrets ==='
source "$CREDS_FILE"
if [[ -n "${PHYAGI_API_KEY:-}" ]]; then
  export OM2W_JUDGE_API_KEY="${OM2W_JUDGE_API_KEY:-$PHYAGI_API_KEY}"
  export OPENAI_GATEWAY_API_KEY="$PHYAGI_API_KEY"
elif [[ -n "${OPENAI_GATEWAY_API_KEY:-}" ]]; then
  export OM2W_JUDGE_API_KEY="${OM2W_JUDGE_API_KEY:-$OPENAI_GATEWAY_API_KEY}"
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  export OM2W_JUDGE_API_KEY="${OM2W_JUDGE_API_KEY:-$OPENAI_API_KEY}"
fi
export OPENAI_GATEWAY_ENDPOINT="${OPENAI_GATEWAY_ENDPOINT:-http://gateway.phyagi.net/api/responses}"

export REPO
export PY=python
export CKPT="$EVAL_CKPT"
export MODEL_NAME="${MODEL_NAME:-$(basename "$EVAL_CKPT")}"
export BENCHMARK_CONFIG
export OUTPUT_DIR
export TASK_LEVEL
export LIMIT
export WORKERS
export JUDGE_RUNS
export TP
export MAX_MODEL_LEN
export MAX_OUTPUT_TOKENS
export GPU_MEMORY_UTILIZATION
export VLLM_LOG_FILE="$OUTPUT_DIR/vllm.log"
export VLLM_LOG_TO_STDOUT="${VLLM_LOG_TO_STDOUT:-0}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_DEBUG="${NCCL_DEBUG_OVERRIDE:-WARN}"

echo '[mini-harness] === GPU preflight ==='
nvidia-smi -L
python - <<'PY'
import torch
import vllm
print("torch", torch.__version__, "vllm", vllm.__version__)
PY

EXTRA_CONFIGS=(
  "run.judge_python=python"
  "run.judge_script=$REPO/om2w_judge/run.py"
  "run.logs_root=$LOG_ROOT"
  "environment.credentials_file=$CREDS_FILE"
)
EXTRA_CONFIGS_CSV=$(IFS=,; echo "${EXTRA_CONFIGS[*]}")
export EXTRA_CONFIGS="$EXTRA_CONFIGS_CSV"

echo '[mini-harness] === launch historical harness ==='
exec bash "$REPO/scripts/mini_harness_eval_sft_vllm.sh" \
  --judge-python python \
  --judge-script "$REPO/om2w_judge/run.py" \
  --judge-num-proc "$JUDGE_NUM_PROC"