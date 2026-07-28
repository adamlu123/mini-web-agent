#!/usr/bin/env bash
# 多节点数据并行 + 断点续评的 OM2W harness eval 驱动(mini-web-agent 自带 harness,
# 非 SkyRL)。每个节点(pod)各跑一份本脚本:
#   1. 本节点起一个 vLLM serve(tp=TP,默认吃满本节点 8 卡)加载 EVAL_CKPT;
#   2. 跑 miniswewebagent.run.benchmarks.om2w 的第 RANK 个 shard
#      (--num-shards WORLD_SIZE --shard-index RANK --resume),任务目录写到
#      PVC 共享 run 目录,天然断点续评:同一 EVAL_RUN_ID 重提即跳过已完成任务;
#   3. master(rank 0)等所有 shard 落 done 标记后,对整个输出目录跑 judge
#      (--judge-only)并写最终汇总(总分 + 按 level breakdown)。
#
# 共享布局(PVC,重提同 EVAL_RUN_ID 即续):
#   $DATA_ROOT/evals/$EVAL_RUN_ID/
#     outputs/<task_id>/result.json   逐任务产物(resume 的判据)
#     outputs_eval_<i>/               judge 结果
#     logs/                           batch/generation/judge 日志与汇总
#     shards/shard_<k>_of_<n>.done    barrier 标记(内容=JOB_NAME rc=<rc>,
#                                     旧 job 的残留标记因 JOB_NAME 不同不会误触发)
#
# 必需 env(submit_job.sh 自动注入):DATA_ROOT JOB_NAME
# 多节点 env(volcano pytorch 插件注入,单节点缺省 1/0):WORLD_SIZE RANK
# 业务 env(由 submit_dist_eval_q35_image.sh 转发,或被 SFT 链设置):
#   EVAL_CKPT     HF 格式权重目录(必需)
#   EVAL_RUN_ID   本次 eval 的稳定 id(必需;重提同 id = 断点续评)
#   BENCHMARK_CONFIG TASK_LEVEL LIMIT WORKERS JUDGE_RUNS JUDGE_NUM_PROC
#   TP MAX_MODEL_LEN MAX_OUTPUT_TOKENS MAX_CONTEXT_TOKENS
#   SLIDING_WINDOW_KEEP_TURNS GPU_MEMORY_UTILIZATION
#   MODEL_NAME CHAT_TEMPLATE RETRY_FAILED EXTRA_CONFIGS JUDGE_ENDPOINT
#   REQUIRE_RUNTIME_MANIFEST ALLOW_TRAINING_CONTRACT_OVERRIDE
#   EVAL_CONTRACT_PREFLIGHT_ONLY CANONICAL_REPO_LINK
#   REPO(SFT 链复用时指到 $CODE_ROOT/mini-web-agent)
# secrets:/run/secrets/webchain-sampling/cred.sh(browserbase + judge key)

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT not set}"
: "${JOB_NAME:?JOB_NAME not set}"
: "${EVAL_CKPT:?EVAL_CKPT not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID not set (stable id; resubmit the same id to resume)}"

NNODES="${EVAL_NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${EVAL_NODE_RANK:-${RANK:-0}}"
IS_MASTER=0; [[ "$NODE_RANK" == "0" ]] && IS_MASTER=1

UPLOAD_ROOT="$DATA_ROOT/runs/$JOB_NAME"
REPO="${REPO:-$UPLOAD_ROOT/mini-web-agent}"
# $REPO 在共享 PVC 上、且同 job 的 N 个 pod 会并发 pip install -e(往源码树里
# 写 egg-info/pth 会互相打架),所以先整棵拷到 pod 本地再装/运行;也顺便让
# import 走本地盘。
if [[ -d "$REPO" ]]; then
  LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-evalcopy}"
  rm -rf "$LOCAL_REPO"
  mkdir -p "$LOCAL_REPO"
  # 不能 cp -a:PVC -> 容器 /tmp(overlayfs)保留权限会 "Operation not
  # supported",在 set -e 下直接打挂 driver(29a12 的死因)
  cp -R --no-preserve=mode,ownership,timestamps "$REPO/." "$LOCAL_REPO/"
  REPO="$LOCAL_REPO"
fi
CANONICAL_REPO_LINK="${CANONICAL_REPO_LINK:-/home/luyadong/sandbox/mini-web-agent}"
mkdir -p "$(dirname "$CANONICAL_REPO_LINK")"
ln -sfnT "$REPO" "$CANONICAL_REPO_LINK"
ENV_ROOT="$DATA_ROOT/envs/q35-mini-harness"
REQ="$REPO/docker/requirements.txt"
MISSING="$ENV_ROOT/requirements.missing.rank${NODE_RANK}.txt"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"

RUN_ROOT="$DATA_ROOT/evals/$EVAL_RUN_ID"
OUTPUTS_DIR="$RUN_ROOT/outputs"
LOGS_DIR="$RUN_ROOT/logs"
SHARDS_DIR="$RUN_ROOT/shards"
DONE_FILE="$SHARDS_DIR/shard_${NODE_RANK}_of_${NNODES}.done"

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark/om2w_sft_state_debug_vllm_sft_ckpt.yaml}"
TASK_LEVEL="${TASK_LEVEL:-all}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-20}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
JUDGE_SCORE_THRESHOLD="${JUDGE_SCORE_THRESHOLD:-3}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://gateway.phyagi.net/api/responses}"
TP="${TP:-8}"
# Manifest-bearing checkpoints own these defaults. Explicit env values are
# treated as requested overrides and validated before GPU startup.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-}"
SLIDING_WINDOW_KEEP_TURNS="${SLIDING_WINDOW_KEEP_TURNS:-}"
ALLOW_TRAINING_CONTRACT_OVERRIDE="${ALLOW_TRAINING_CONTRACT_OVERRIDE:-0}"
REQUIRE_RUNTIME_MANIFEST="${REQUIRE_RUNTIME_MANIFEST:-0}"
EVAL_CONTRACT_PREFLIGHT_ONLY="${EVAL_CONTRACT_PREFLIGHT_ONLY:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MODEL_NAME="${MODEL_NAME:-$(basename "$EVAL_CKPT")}"
TASKS_FILE="${TASKS_FILE:-$REPO/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
# Manifest checkpoints use the exact template bundled by PhiTrain.
CHAT_TEMPLATE_WAS_SET=0
[[ -v CHAT_TEMPLATE ]] && CHAT_TEMPLATE_WAS_SET=1
CHAT_TEMPLATE="${CHAT_TEMPLATE-}"
# CHAT_TEMPLATE_NAME:相对 $REPO/configs 的模板文件名(跨机路径无关的指定方式)
[[ -n "${CHAT_TEMPLATE_NAME:-}" ]] && { CHAT_TEMPLATE="$REPO/configs/$CHAT_TEMPLATE_NAME"; CHAT_TEMPLATE_WAS_SET=1; }
HOST=127.0.0.1
PORT="${PORT:-8000}"
ENDPOINT="http://${HOST}:${PORT}/v1/chat/completions"
# master 等其它 shard 的超时(秒)
EVAL_BARRIER_TIMEOUT="${EVAL_BARRIER_TIMEOUT:-14400}"

echo "[dist-eval] job=$JOB_NAME host=$(hostname) shard=$NODE_RANK/$NNODES master=$IS_MASTER"
echo "[dist-eval] run_id=$EVAL_RUN_ID run_root=$RUN_ROOT"
echo "[dist-eval] ckpt=$EVAL_CKPT model=$MODEL_NAME"
echo "[dist-eval] benchmark=$BENCHMARK_CONFIG level=$TASK_LEVEL limit=$LIMIT workers/node=$WORKERS"
echo "[dist-eval] judge_model=$JUDGE_MODEL judge_workers=$JUDGE_NUM_PROC endpoint=${JUDGE_ENDPOINT:-<openai>}"

[[ -d "$REPO" ]] || { echo "[dist-eval][error] repo not found: $REPO"; exit 1; }
[[ -f "$CREDS_FILE" ]] || { echo "[dist-eval][error] credentials file not found: $CREDS_FILE"; exit 1; }
[[ -f "$TASKS_FILE" ]] || { echo "[dist-eval][error] tasks file not found: $TASKS_FILE"; exit 1; }
[[ -f "$REPO/scripts/eval_with_original_om2w.py" ]] || {
  echo "[dist-eval][error] original OM2W evaluator not found"; exit 1; }
[[ -d "$EVAL_CKPT" ]] || { echo "[dist-eval][error] ckpt dir not found: $EVAL_CKPT"; exit 1; }
if ! compgen -G "$EVAL_CKPT/*.safetensors" >/dev/null && [[ ! -f "$EVAL_CKPT/model.safetensors.index.json" ]]; then
  echo "[dist-eval][error] ckpt is not HF safetensors format: $EVAL_CKPT"; exit 1
fi
[[ "$REQUIRE_RUNTIME_MANIFEST" == "0" ||
   "$REQUIRE_RUNTIME_MANIFEST" == "1" ]] || {
  echo "[dist-eval][error] REQUIRE_RUNTIME_MANIFEST must be 0 or 1" >&2
  exit 1
}
[[ "$ALLOW_TRAINING_CONTRACT_OVERRIDE" == "0" ||
   "$ALLOW_TRAINING_CONTRACT_OVERRIDE" == "1" ]] || {
  echo "[dist-eval][error] ALLOW_TRAINING_CONTRACT_OVERRIDE must be 0 or 1" >&2
  exit 1
}
[[ "$EVAL_CONTRACT_PREFLIGHT_ONLY" == "0" ||
   "$EVAL_CONTRACT_PREFLIGHT_ONLY" == "1" ]] || {
  echo "[dist-eval][error] EVAL_CONTRACT_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 1
}
RUNTIME_MANIFEST="$EVAL_CKPT/web_agent_runtime.json"
if [[ "$REQUIRE_RUNTIME_MANIFEST" == "1" && ! -f "$RUNTIME_MANIFEST" ]]; then
  echo "[dist-eval][error] checkpoint is missing required $RUNTIME_MANIFEST" >&2
  exit 1
fi
mkdir -p "$ENV_ROOT" "$OUTPUTS_DIR" "$LOGS_DIR" "$SHARDS_DIR"
# 清掉本 shard 的历史 done 标记(barrier 只认内容含当前 JOB_NAME 的标记,
# 这里删除只是保持目录干净)
rm -f "$DONE_FILE"

if [[ "${EVAL_SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  echo '[dist-eval] === install missing python deps without touching image CUDA/torch/vLLM stack ==='
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

print(f"[dist-eval] excluded compiled-stack reqs: {len(excluded)}")
print(f"[dist-eval] already installed reqs: {len(skipped)}")
print(f"[dist-eval] installing reqs: {len(kept)} forced={forced}")
PY

  if [[ -s "$MISSING" ]]; then
    pip install --no-deps -r "$MISSING"
  else
    echo '[dist-eval] no missing deps'
  fi
  pip install --no-deps --no-build-isolation -e "$REPO"
  python -c "import openai, rich, typer, browserbase, miniswewebagent; print('[dist-eval] import preflight OK')"
fi

echo '[dist-eval] === resolve WebWright training/runtime contract ==='
RESOLVED_RUNTIME="$LOGS_DIR/resolved_runtime_shard${NODE_RANK}.json"
HAS_RUNTIME_MANIFEST=0
if [[ -f "$RUNTIME_MANIFEST" ]]; then
  HAS_RUNTIME_MANIFEST=1
  VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
  RUNTIME_ARGS=(
    preflight
    --checkpoint "$EVAL_CKPT"
    --output "$RESOLVED_RUNTIME"
    --vllm-version "$VLLM_VERSION"
  )
  [[ -n "$MAX_MODEL_LEN" ]] && RUNTIME_ARGS+=( --max-model-len "$MAX_MODEL_LEN" )
  [[ -n "$MAX_CONTEXT_TOKENS" ]] &&
    RUNTIME_ARGS+=( --max-context-tokens "$MAX_CONTEXT_TOKENS" )
  [[ -n "$MAX_OUTPUT_TOKENS" ]] &&
    RUNTIME_ARGS+=( --max-output-tokens "$MAX_OUTPUT_TOKENS" )
  [[ -n "$SLIDING_WINDOW_KEEP_TURNS" ]] &&
    RUNTIME_ARGS+=( --sliding-window-keep-turns "$SLIDING_WINDOW_KEEP_TURNS" )
  [[ "$ALLOW_TRAINING_CONTRACT_OVERRIDE" == "1" ]] &&
    RUNTIME_ARGS+=( --allow-training-contract-override )

  PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    python -m miniswewebagent.utils.web_agent_runtime "${RUNTIME_ARGS[@]}" \
    >"$LOGS_DIR/runtime_preflight_shard${NODE_RANK}.log"

  mapfile -t RUNTIME_VALUES < <(
    python - "$RESOLVED_RUNTIME" <<'PY'
import json
import sys
from pathlib import Path

runtime = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
inference = runtime["resolved_inference"]
processor = runtime["processor"]
values = (
    runtime["contract_id"],
    inference["max_model_len"],
    inference["max_context_tokens"],
    inference["max_output_tokens"],
    inference["sliding_window_keep_turns"],
    runtime["resolved_model_config"]["text_only_image_policy"],
    json.dumps(runtime["resolved_model_config"]["stop_sequences"], separators=(",", ":")),
    processor["min_pixels"],
    processor["max_pixels"],
    str(Path(runtime["checkpoint_path"]) / runtime["chat_template"]["file"]),
)
for value in values:
    print(value)
PY
  )
  if (( ${#RUNTIME_VALUES[@]} != 10 )); then
    echo "[dist-eval][error] runtime preflight returned ${#RUNTIME_VALUES[@]} values; expected 10" >&2
    exit 1
  fi
  CONTRACT_ID="${RUNTIME_VALUES[0]}"
  MAX_MODEL_LEN="${RUNTIME_VALUES[1]}"
  MAX_CONTEXT_TOKENS="${RUNTIME_VALUES[2]}"
  MAX_OUTPUT_TOKENS="${RUNTIME_VALUES[3]}"
  SLIDING_WINDOW_KEEP_TURNS="${RUNTIME_VALUES[4]}"
  TEXT_ONLY_IMAGE_POLICY="${RUNTIME_VALUES[5]}"
  STOP_SEQUENCES_JSON="${RUNTIME_VALUES[6]}"
  MIN_PIXELS="${RUNTIME_VALUES[7]}"
  MAX_PIXELS="${RUNTIME_VALUES[8]}"
  CONTRACT_CHAT_TEMPLATE="${RUNTIME_VALUES[9]}"
  if [[ -n "$CHAT_TEMPLATE" ]] &&
      [[ "$(realpath -m "$CHAT_TEMPLATE")" != "$(realpath -m "$CONTRACT_CHAT_TEMPLATE")" ]] &&
      [[ "$ALLOW_TRAINING_CONTRACT_OVERRIDE" != "1" ]]; then
    echo "[dist-eval][error] CHAT_TEMPLATE conflicts with manifest template $CONTRACT_CHAT_TEMPLATE" >&2
    exit 1
  fi
  CHAT_TEMPLATE="${CHAT_TEMPLATE:-$CONTRACT_CHAT_TEMPLATE}"
else
  # Compatibility for the existing LlamaFactory Mode A chain and legacy
  # already-merged Mode B checkpoints. These checkpoints predate the PhiTrain
  # runtime manifest, so preserve their previous launcher defaults rather than
  # silently treating them as WebWright-aligned.
  echo "[dist-eval][warning] no runtime manifest; using unvalidated manifestless compatibility defaults" >&2
  echo "[dist-eval][warning] checkpoint must already contain the full vision model" >&2
  echo "[dist-eval][warning] set REQUIRE_RUNTIME_MANIFEST=1 for PhiTrain WebWright checkpoints" >&2
  CONTRACT_ID="manifestless_compat"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
  MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-48000}"
  MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
  SLIDING_WINDOW_KEEP_TURNS="${SLIDING_WINDOW_KEEP_TURNS:-10}"
  TEXT_ONLY_IMAGE_POLICY="none"
  STOP_SEQUENCES_JSON="[]"
  if [[ "$CHAT_TEMPLATE_WAS_SET" != "1" ]]; then
    CHAT_TEMPLATE="$REPO/configs/qwen3_5_train_aligned.jinja"
  fi
fi

for numeric_name in \
  MAX_MODEL_LEN MAX_CONTEXT_TOKENS MAX_OUTPUT_TOKENS \
  SLIDING_WINDOW_KEEP_TURNS; do
  numeric_value="${!numeric_name}"
  [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "[dist-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
    exit 1
  }
done
STOP_SEQUENCES_JSON="$(
  python - "$STOP_SEQUENCES_JSON" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
if not isinstance(value, list) or not all(
    isinstance(item, str) and item for item in value
):
    raise SystemExit("stop sequences must be a JSON list of non-empty strings")
print(json.dumps(value, separators=(",", ":")))
PY
)"
VLLM_MM_ARGS=()
MM_PROCESSOR_DESCRIPTION="<checkpoint defaults>"
if [[ "$HAS_RUNTIME_MANIFEST" == "1" ]]; then
  for numeric_name in MIN_PIXELS MAX_PIXELS; do
    numeric_value="${!numeric_name}"
    [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
      echo "[dist-eval][error] $numeric_name must be a positive integer: $numeric_value" >&2
      exit 1
    }
  done
  [[ "$TEXT_ONLY_IMAGE_POLICY" == "black_56" ]] || {
    echo "[dist-eval][error] text-only image policy must be black_56: $TEXT_ONLY_IMAGE_POLICY" >&2
    exit 1
  }
  [[ "$STOP_SEQUENCES_JSON" != "[]" ]] || {
    echo "[dist-eval][error] manifest stop sequences must not be empty" >&2
    exit 1
  }
  (( MIN_PIXELS <= MAX_PIXELS )) || {
    echo "[dist-eval][error] MIN_PIXELS exceeds MAX_PIXELS" >&2
    exit 1
  }
  MM_PROCESSOR_KWARGS="{\"min_pixels\":${MIN_PIXELS},\"max_pixels\":${MAX_PIXELS},\"use_fast\":false}"
  VLLM_MM_ARGS=( --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" )
  MM_PROCESSOR_DESCRIPTION="$MM_PROCESSOR_KWARGS"
elif [[ "$TEXT_ONLY_IMAGE_POLICY" != "none" ]]; then
  echo "[dist-eval][error] manifestless compatibility requires text-only image policy none" >&2
  exit 1
fi
if (( MAX_CONTEXT_TOKENS + MAX_OUTPUT_TOKENS > MAX_MODEL_LEN )); then
  echo "[dist-eval][error] context budget + output exceeds max model length: $MAX_CONTEXT_TOKENS + $MAX_OUTPUT_TOKENS > $MAX_MODEL_LEN" >&2
  exit 1
fi
[[ -z "$CHAT_TEMPLATE" || -f "$CHAT_TEMPLATE" ]] || {
  echo "[dist-eval][error] resolved chat template not found: $CHAT_TEMPLATE" >&2
  exit 1
}
echo "[dist-eval] contract=$CONTRACT_ID max_len=$MAX_MODEL_LEN input_budget=$MAX_CONTEXT_TOKENS output=$MAX_OUTPUT_TOKENS"
echo "[dist-eval] template=$CHAT_TEMPLATE image_policy=$TEXT_ONLY_IMAGE_POLICY stop_sequences=$STOP_SEQUENCES_JSON mm_processor=$MM_PROCESSOR_DESCRIPTION"
if [[ "$EVAL_CONTRACT_PREFLIGHT_ONLY" == "1" ]]; then
  echo "[dist-eval] contract preflight complete; exiting before credentials/GPU startup"
  exit 0
fi

echo '[dist-eval] === source secrets ==='
source "$CREDS_FILE"
if [[ -n "${PHYAGI_API_KEY:-}" ]]; then
  export OM2W_JUDGE_API_KEY="${OM2W_JUDGE_API_KEY:-$PHYAGI_API_KEY}"
  export OPENAI_GATEWAY_API_KEY="$PHYAGI_API_KEY"
elif [[ -n "${OPENAI_GATEWAY_API_KEY:-}" ]]; then
  export OM2W_JUDGE_API_KEY="${OM2W_JUDGE_API_KEY:-$OPENAI_GATEWAY_API_KEY}"
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  export OM2W_JUDGE_API_KEY="${OM2W_JUDGE_API_KEY:-$OPENAI_API_KEY}"
fi

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_DEBUG="${NCCL_DEBUG_OVERRIDE:-WARN}"

echo '[dist-eval] === GPU preflight ==='
nvidia-smi -L

echo "[dist-eval] === vllm serve $MODEL_NAME (tp=$TP, max_len=$MAX_MODEL_LEN) ==="
VLLM_LOG_FILE="$LOGS_DIR/vllm_shard${NODE_RANK}.log"
VLLM_TEMPLATE_ARGS=()
[[ -n "$CHAT_TEMPLATE" ]] && VLLM_TEMPLATE_ARGS=( --chat-template "$CHAT_TEMPLATE" )
vllm serve "$EVAL_CKPT" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${VLLM_MM_ARGS[@]}" \
  --trust-remote-code \
  "${VLLM_TEMPLATE_ARGS[@]}" >"$VLLM_LOG_FILE" 2>&1 &
VLLM_PID=$!
cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if ! python - "$VLLM_PID" <<PY
import sys, time, urllib.request
from pathlib import Path

pid = int(sys.argv[1])
url = "http://${HOST}:${PORT}/v1/models"
deadline = time.time() + int("${VLLM_WAIT_SECONDS:-1800}")
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print("[dist-eval] vllm ready")
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists() or stat_path.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            print("[dist-eval][error] vLLM exited before becoming ready", file=sys.stderr)
            raise SystemExit(1)
        time.sleep(5)
print(f"[dist-eval][error] vLLM not ready before deadline: {url}", file=sys.stderr)
raise SystemExit(1)
PY
then
  tail -200 "$VLLM_LOG_FILE" >&2
  exit 1
fi

echo "[dist-eval] === shard $NODE_RANK/$NNODES generation (resume on) ==="
cd "$REPO"
CONFIG_ARGS=( -c "$BENCHMARK_CONFIG" )
if [[ -n "${EXTRA_CONFIGS:-}" ]]; then
  IFS=',' read -r -a _extra_cfgs <<< "$EXTRA_CONFIGS"
  for _cfg in "${_extra_cfgs[@]}"; do
    _cfg="${_cfg#${_cfg%%[![:space:]]*}}"; _cfg="${_cfg%${_cfg##*[![:space:]]}}"
    [[ -n "$_cfg" ]] && CONFIG_ARGS+=( -c "$_cfg" )
  done
fi
RETRY_ARGS=()
[[ "${RETRY_FAILED:-0}" == "1" ]] && RETRY_ARGS=( --retry-failed )

GEN_RC=0
python -m miniswewebagent.run.benchmarks.om2w \
  "${CONFIG_ARGS[@]}" \
  -c "model.endpoint=$ENDPOINT" \
  -c "model.model_name=$MODEL_NAME" \
  -c "model.max_output_tokens=$MAX_OUTPUT_TOKENS" \
  -c "model.max_context_tokens=$MAX_CONTEXT_TOKENS" \
  -c "model.sliding_window_keep_turns=$SLIDING_WINDOW_KEEP_TURNS" \
  -c "model.text_only_image_policy=$TEXT_ONLY_IMAGE_POLICY" \
  -c "model.stop_sequences=$STOP_SEQUENCES_JSON" \
  -c "environment.env.WEB_AGENT_POLICY_URL=$ENDPOINT" \
  -c "environment.env.WEB_AGENT_POLICY_MODEL=$MODEL_NAME" \
  -c "environment.env.OPENAI_COMPATIBLE_ENDPOINT=$ENDPOINT" \
  -c "environment.env.OPENAI_COMPATIBLE_MODEL=$MODEL_NAME" \
  -c "environment.env.OPENAI_COMPATIBLE_API_KEY=dummy" \
  -c "environment.env.OPENAI_GATEWAY_MODEL=$MODEL_NAME" \
  -c "run.logs_root=$LOGS_DIR" \
  -c "environment.credentials_file=$CREDS_FILE" \
  --task-level "$TASK_LEVEL" \
  --limit "$LIMIT" \
  --workers "$WORKERS" \
  --num-shards "$NNODES" \
  --shard-index "$NODE_RANK" \
  --resume "${RETRY_ARGS[@]}" \
  --batch-name "$EVAL_RUN_ID" \
  --no-evaluate \
  --output-dir "$OUTPUTS_DIR" || GEN_RC=$?
echo "[dist-eval] shard $NODE_RANK generation rc=$GEN_RC"

# vLLM 在 judge 阶段不需要,尽早释放显存/让出节点
cleanup; VLLM_PID=""

echo "$JOB_NAME rc=$GEN_RC" > "$DONE_FILE"

if [[ "$IS_MASTER" != "1" ]]; then
  # Volcano job 策略是 TaskCompleted->CompleteJob、PodFailed->AbortJob:
  # worker 提前退出会把整个 job 判完、杀掉还在 judge 的 master;非零退出会
  # Abort 全 job。所以 worker 记录 rc 后原地等 master 的完成标记,并恒以 0
  # 退出——shard 失败由 master 从 done 文件聚合、反映在 master 的退出码上。
  COMPLETE_FILE="$SHARDS_DIR/job_complete.$JOB_NAME"
  echo "[dist-eval] [worker $NODE_RANK] shard done (rc=$GEN_RC); waiting for master judge ($COMPLETE_FILE)"
  deadline=$(( $(date +%s) + EVAL_BARRIER_TIMEOUT ))
  while [[ ! -f "$COMPLETE_FILE" ]] && (( $(date +%s) < deadline )); do sleep 30; done
  [[ -f "$COMPLETE_FILE" ]] || echo "[dist-eval][warn] [worker $NODE_RANK] timed out waiting for master; exiting anyway"
  exit 0
fi

echo "[dist-eval] === [master] barrier: waiting for $NNODES shard(s), timeout ${EVAL_BARRIER_TIMEOUT}s ==="
SHARD_RCS=()
deadline=$(( $(date +%s) + EVAL_BARRIER_TIMEOUT ))
while :; do
  SHARD_RCS=()
  all_done=1
  for k in $(seq 0 $((NNODES - 1))); do
    f="$SHARDS_DIR/shard_${k}_of_${NNODES}.done"
    # 只认内容带当前 JOB_NAME 的标记,旧 job 的残留不算数
    if [[ -f "$f" ]] && grep -q "$JOB_NAME" "$f"; then
      SHARD_RCS+=( "$(sed -n 's/.*rc=//p' "$f" | head -1)" )
    else
      all_done=0
    fi
  done
  [[ "$all_done" == "1" ]] && break
  if (( $(date +%s) > deadline )); then
    echo "[dist-eval][warn] barrier timeout; judging whatever is in $OUTPUTS_DIR (missing tasks score as fail)"
    break
  fi
  sleep 30
done
echo "[dist-eval] [master] shard rcs: ${SHARD_RCS[*]:-<incomplete>}"

GENERATION_FAILURES="$(
  python - "$OUTPUTS_DIR" <<'PY'
import json
import sys
from pathlib import Path

failures = 0
for result_path in Path(sys.argv[1]).glob("*/result.json"):
    try:
        failures += bool(json.loads(result_path.read_text(encoding="utf-8")).get("run_exception"))
    except Exception:
        failures += 1
print(failures)
PY
)"
if (( GENERATION_FAILURES > 0 )); then
  echo "[dist-eval][warn] generation left $GENERATION_FAILURES retryable task failure(s)"
fi

JUDGE_RC=0
if [[ "${PERSISTENT_JUDGE:-0}" == "1" ]]; then
  # SPB(persistent-browser CLI)轨迹判分:动作史=browser-steps.jsonl 的 action,
  # 截图=workspace/screenshots/ 根目录 PNG。走 om2w_judge.utils.OpenaiEngine,
  # JUDGE_ENDPOINT 非空则经 gateway(用 OPENAI_GATEWAY_API_KEY/PHYAGI_API_KEY)。
  echo "[dist-eval] === [master] persistent-CLI OM2W judge (eval_persistent_cli_with_original_om2w.py) ==="
  PERS_OUT="$RUN_ROOT/outputs_eval_persistent"
  # --expected_tasks 0:关闭数量硬校验(分片缺任务时照判,缺失反映在 summary 里)
  python "$REPO/scripts/eval_persistent_cli_with_original_om2w.py" \
    --trajectories_dir "$OUTPUTS_DIR" \
    --output_path "$PERS_OUT" \
    --model "$JUDGE_MODEL" \
    --score_threshold "$JUDGE_SCORE_THRESHOLD" \
    --num_worker "$JUDGE_NUM_PROC" \
    --expected_tasks 0 \
    --endpoint_target_uri "${JUDGE_ENDPOINT:-}" || JUDGE_RC=$?
  echo "[dist-eval] persistent judge rc=$JUDGE_RC ; summary: $PERS_OUT/eval_summary.json"
elif [[ "${ORIGINAL_JUDGE:-0}" == "1" ]]; then
  # 原版 judge:scripts/eval_with_original_om2w.py(plain_text 全量动作日志)。
  # 仓库版 OpenaiEngine 恒直连 api.openai.com,须用 cred.sh 的 OPENAI_API_KEY
  # 且不能让 gateway 变量/BACKUP_KEY 干扰。
  echo "[dist-eval] === [master] original OM2W judge (eval_with_original_om2w.py) ==="
  : "${OPENAI_API_KEY:?ORIGINAL_JUDGE=1 needs OPENAI_API_KEY from cred.sh}"
  ORIG_SUMMARY="$LOGS_DIR/$EVAL_RUN_ID/run_summary_original_judge.json"
  env -u OPENAI_GATEWAY_ENDPOINT -u OPENAI_GATEWAY_API_KEY -u OPENAI_API_BACKUP_KEY \
    python "$REPO/scripts/eval_with_original_om2w.py" \
    --trajectories_dir "$OUTPUTS_DIR" \
    --output_path "$RUN_ROOT/outputs_eval_original" \
    --tasks_file "$TASKS_FILE" \
    --model "$JUDGE_MODEL" \
    --api_key "$OPENAI_API_KEY" \
    --endpoint_target_uri "" \
    --score_threshold "$JUDGE_SCORE_THRESHOLD" \
    --task_level "$TASK_LEVEL" \
    --limit "$LIMIT" \
    --num_worker "$JUDGE_NUM_PROC" \
    --summary_path "$ORIG_SUMMARY" || JUDGE_RC=$?
  echo "[dist-eval] original judge rc=$JUDGE_RC ; final summary: $ORIG_SUMMARY"
else
  echo "[dist-eval] === [master] native harness judge over full output dir ==="
  JUDGE_ENDPOINT_ARGS=()
  [[ -n "$JUDGE_ENDPOINT" ]] && JUDGE_ENDPOINT_ARGS=( --judge-endpoint "$JUDGE_ENDPOINT" )
  python -m miniswewebagent.run.benchmarks.om2w \
    "${CONFIG_ARGS[@]}" \
    -c "run.logs_root=$LOGS_DIR" \
    -c "environment.credentials_file=$CREDS_FILE" \
    --task-level "$TASK_LEVEL" \
    --limit "$LIMIT" \
    --judge-only \
    --batch-name "$EVAL_RUN_ID" \
    --judge-runs "$JUDGE_RUNS" \
    --judge-num-proc "$JUDGE_NUM_PROC" \
    --judge-python python \
    --judge-script "$REPO/om2w_judge_sandbox/run.py" \
    "${JUDGE_ENDPOINT_ARGS[@]}" \
    --output-dir "$OUTPUTS_DIR" || JUDGE_RC=$?
  echo "[dist-eval] judge rc=$JUDGE_RC ; final summary: $LOGS_DIR/$EVAL_RUN_ID/run_summary_judge.json"
fi

FINAL_RC=$JUDGE_RC
(( GENERATION_FAILURES > 0 )) && FINAL_RC=1
for rc in "${SHARD_RCS[@]:-}"; do
  [[ -n "$rc" && "$rc" != "0" ]] && FINAL_RC=1
done
# 放行还在等待的 worker(见 worker 分支的注释)
echo "rc=$FINAL_RC" > "$SHARDS_DIR/job_complete.$JOB_NAME"
exit "$FINAL_RC"
