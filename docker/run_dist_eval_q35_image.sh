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
#   $PVC_MOUNT/$USER_ALIAS/evals/$EVAL_RUN_ID/
#     outputs/<task_id>/result.json   逐任务产物(resume 的判据)
#     outputs_eval_<i>/               judge 结果
#     logs/                           batch/generation/judge 日志与汇总
#     shards/shard_<k>_of_<n>.done    barrier 标记(内容=JOB_NAME rc=<rc>,
#                                     旧 job 的残留标记因 JOB_NAME 不同不会误触发)
#
# 必需 env(submit_job.sh 自动注入):PVC_MOUNT USER_ALIAS JOB_NAME
# 多节点 env(volcano pytorch 插件注入,单节点缺省 1/0):WORLD_SIZE RANK
# 业务 env(由 submit_dist_eval_q35_image.sh 转发,或被 SFT 链设置):
#   EVAL_CKPT     HF 格式权重目录(必需)
#   EVAL_RUN_ID   本次 eval 的稳定 id(必需;重提同 id = 断点续评)
#   BENCHMARK_CONFIG TASK_LEVEL LIMIT WORKERS JUDGE_RUNS JUDGE_NUM_PROC
#   TP MAX_MODEL_LEN MAX_OUTPUT_TOKENS MAX_CONTEXT_TOKENS GPU_MEMORY_UTILIZATION
#   MODEL_NAME CHAT_TEMPLATE RETRY_FAILED EXTRA_CONFIGS JUDGE_ENDPOINT
#   REPO(SFT 链复用时指到 $CODE_ROOT/mini-web-agent)
# secrets:/run/secrets/webchain-sampling/cred.sh(browserbase + judge key)

set -euo pipefail

: "${PVC_MOUNT:?PVC_MOUNT not set}"
: "${USER_ALIAS:?USER_ALIAS not set}"
: "${JOB_NAME:?JOB_NAME not set}"
: "${EVAL_CKPT:?EVAL_CKPT not set}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID not set (stable id; resubmit the same id to resume)}"

NNODES="${EVAL_NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${EVAL_NODE_RANK:-${RANK:-0}}"
IS_MASTER=0; [[ "$NODE_RANK" == "0" ]] && IS_MASTER=1

UPLOAD_ROOT="$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME"
REPO="${REPO:-$UPLOAD_ROOT/mini-web-agent}"
# $REPO 在共享 PVC 上、且同 job 的 N 个 pod 会并发 pip install -e(往源码树里
# 写 egg-info/pth 会互相打架),所以先整棵拷到 pod 本地再装/运行;也顺便让
# import 走本地盘。
if [[ -d "$REPO" ]]; then
  LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-evalcopy}"
  rm -rf "$LOCAL_REPO"
  cp -a "$REPO" "$LOCAL_REPO"
  REPO="$LOCAL_REPO"
fi
ENV_ROOT="$PVC_MOUNT/$USER_ALIAS/envs/q35-mini-harness"
REQ="$REPO/docker/requirements.txt"
MISSING="$ENV_ROOT/requirements.missing.rank${NODE_RANK}.txt"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"

RUN_ROOT="$PVC_MOUNT/$USER_ALIAS/evals/$EVAL_RUN_ID"
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
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
# 客户端 sliding window 预算(openrouter_model.py;0=关)。48000 是 65536 上下文
# 下留出估算容错的验证值,防长会话 vLLM 400。
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-48000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MODEL_NAME="${MODEL_NAME:-$(basename "$EVAL_CKPT")}"
# 训练对齐的 chat template(qwen3_5 think 对齐,docs/qwen3_5_think_alignment.md)。
# ckpt 自带的 chat_template.jinja 与它不同,不能省。置空则用 ckpt 自带模板。
CHAT_TEMPLATE="${CHAT_TEMPLATE-$REPO/configs/qwen3_5_train_aligned.jinja}"
HOST=127.0.0.1
PORT="${PORT:-8000}"
ENDPOINT="http://${HOST}:${PORT}/v1/chat/completions"
# master 等其它 shard 的超时(秒)
EVAL_BARRIER_TIMEOUT="${EVAL_BARRIER_TIMEOUT:-14400}"

echo "[dist-eval] job=$JOB_NAME host=$(hostname) shard=$NODE_RANK/$NNODES master=$IS_MASTER"
echo "[dist-eval] run_id=$EVAL_RUN_ID run_root=$RUN_ROOT"
echo "[dist-eval] ckpt=$EVAL_CKPT model=$MODEL_NAME"
echo "[dist-eval] benchmark=$BENCHMARK_CONFIG level=$TASK_LEVEL limit=$LIMIT workers/node=$WORKERS"

[[ -d "$REPO" ]] || { echo "[dist-eval][error] repo not found: $REPO"; exit 1; }
[[ -f "$CREDS_FILE" ]] || { echo "[dist-eval][error] credentials file not found: $CREDS_FILE"; exit 1; }
[[ -d "$EVAL_CKPT" ]] || { echo "[dist-eval][error] ckpt dir not found: $EVAL_CKPT"; exit 1; }
if ! compgen -G "$EVAL_CKPT/*.safetensors" >/dev/null && [[ ! -f "$EVAL_CKPT/model.safetensors.index.json" ]]; then
  echo "[dist-eval][error] ckpt is not HF safetensors format: $EVAL_CKPT"; exit 1
fi
if [[ -n "$CHAT_TEMPLATE" && ! -f "$CHAT_TEMPLATE" ]]; then
  echo "[dist-eval][error] chat template not found: $CHAT_TEMPLATE (set CHAT_TEMPLATE= to use the ckpt's own)"; exit 1
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

# === 自动 vision merge(评中间 checkpoint-N 时缺 vision tower)================
# 文本 SFT 的中间 ckpt 缺 vision 权重,直接 serve 必挂("visual.* not
# initialized")。缺 vision.safetensors 时:master 把 ckpt 拷到提交者自己的
# models/ 下(不动原目录——可能属于别人且易失)再从 HF 缓存的 base 补全;
# worker 等合并后的目录出现。ckpt 本就完整时零开销。MERGE_VISION=0 关闭。
if [[ ! -f "$EVAL_CKPT/vision.safetensors" && "${MERGE_VISION:-1}" == "1" ]]; then
  MERGED_CKPT="$PVC_MOUNT/$USER_ALIAS/models/evalmerge_$(basename "$(dirname "$EVAL_CKPT")")_$(basename "$EVAL_CKPT")"
  if [[ "$IS_MASTER" == "1" ]]; then
    if [[ ! -f "$MERGED_CKPT/vision.safetensors" ]]; then
      echo "[dist-eval] ckpt lacks vision.safetensors -> copy+merge to $MERGED_CKPT (takes a few min)"
      rm -rf "$MERGED_CKPT.tmp"
      mkdir -p "$(dirname "$MERGED_CKPT")"
      cp -a "$EVAL_CKPT" "$MERGED_CKPT.tmp"
      BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen3.5-9B}"
      HFH="${HF_HOME:-$PVC_MOUNT/$USER_ALIAS/hf_cache}"
      BASE_DIR="$(ls -d "$HFH/hub/models--${BASE_MODEL_ID//\//--}/snapshots/"*/ 2>/dev/null | head -1)"
      [[ -n "$BASE_DIR" ]] || { echo "[dist-eval][error] base snapshot not found under $HFH (need it for vision merge)"; exit 1; }
      python "$REPO/scripts/merge_vision_from_base.py" --ckpt "$MERGED_CKPT.tmp" --base "$BASE_DIR"
      [[ -f "$MERGED_CKPT.tmp/vision.safetensors" ]] || { echo "[dist-eval][error] vision merge failed"; exit 1; }
      mv "$MERGED_CKPT.tmp" "$MERGED_CKPT"
    else
      echo "[dist-eval] reusing existing vision-merged copy: $MERGED_CKPT"
    fi
  else
    echo "[dist-eval] [worker $NODE_RANK] waiting for master's vision-merged ckpt: $MERGED_CKPT"
    for _ in $(seq 1 240); do [[ -f "$MERGED_CKPT/vision.safetensors" ]] && break; sleep 15; done
    [[ -f "$MERGED_CKPT/vision.safetensors" ]] || { echo "[dist-eval][error] timed out waiting for merged ckpt"; exit 1; }
  fi
  EVAL_CKPT="$MERGED_CKPT"
fi

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
  --trust-remote-code \
  "${VLLM_TEMPLATE_ARGS[@]}" ${VLLM_ARGS:-} >"$VLLM_LOG_FILE" 2>&1 &
VLLM_PID=$!
cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

python - <<PY
import sys, time, urllib.request
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
        time.sleep(5)
print(f"[dist-eval][error] vLLM not ready before deadline: {url}", file=sys.stderr)
raise SystemExit(1)
PY

echo "[dist-eval] === shard $NODE_RANK/$NNODES generation (resume on) ==="
cd "$REPO"
CONFIG_ARGS=( -c mini.yaml -c "$BENCHMARK_CONFIG" )
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

echo "[dist-eval] === [master] judge over full output dir ==="
JUDGE_ENDPOINT_ARGS=()
[[ -n "${JUDGE_ENDPOINT:-}" ]] && JUDGE_ENDPOINT_ARGS=( --judge-endpoint "$JUDGE_ENDPOINT" )
JUDGE_RC=0
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
  --judge-script "$REPO/om2w_judge/run.py" \
  "${JUDGE_ENDPOINT_ARGS[@]}" \
  --output-dir "$OUTPUTS_DIR" || JUDGE_RC=$?
echo "[dist-eval] judge rc=$JUDGE_RC ; final summary: $LOGS_DIR/$EVAL_RUN_ID/run_summary_judge.json"

FINAL_RC=$JUDGE_RC
for rc in "${SHARD_RCS[@]:-}"; do
  [[ -n "$rc" && "$rc" != "0" ]] && FINAL_RC=1
done
# 放行还在等待的 worker(见 worker 分支的注释)
echo "rc=$FINAL_RC" > "$SHARDS_DIR/job_complete.$JOB_NAME"
exit "$FINAL_RC"
