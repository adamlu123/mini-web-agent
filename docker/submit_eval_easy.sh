#!/usr/bin/env bash
# Submit an EVAL-ONLY (base performance) web-agent job to bonete61.
#
# Runs echo_rl.web_agent.eval_entrypoint (no training, no checkpoints) over the
# om2w easy *train* + *val* parquets, scoring each separately. The vLLM engine
# loads the base HF weights, so the logged eval/<data_source>/... metrics are
# the untuned base-model performance.
#
# Pick the model with MODEL=4b or MODEL=9b (default 4b):
#   MODEL=4b bash docker/submit_eval_easy.sh
#   MODEL=9b bash docker/submit_eval_easy.sh
#
# IMPORTANT design notes:
#   * The actual in-pod setup lives in docker/run_eval_in_pod.sh (uploaded with
#     the repo). The Volcano job's `--cmd` is just a tiny one-liner that execs
#     it. This is required: the bonete61 API server is behind Cloudflare, whose
#     WAF blocks `kubectl create` POSTs whose body contains a large shell
#     preamble (the `curl ... | sh` uv installer + URLs). Keeping that content
#     out of the request body keeps the create request WAF-clean.
#   * No raw secrets go through `--extra-env-vars`/the YAML. The working sk-proj
#     judge key is delivered via the `echo-rl-openai` k8s secret (base64 in the
#     API request, so also WAF-safe). Create/refresh it once with:
#       kubectl -n bonete61 create secret generic echo-rl-openai \
#         --from-literal=OPENAI_API_KEY="$OPENAI_API_BACKUP_KEY" \
#         --dry-run=client -o yaml | kubectl -n bonete61 apply -f -
#   * Every PVC path is JOB-SCOPED, so the 4B and 9B jobs can run concurrently
#     without racing on a shared CODE_ROOT / venv.
#
# Override config / image / gpus via env:
#   CONFIG=echo_configs/<name>.yaml GPUS=4 IMAGE=... bash docker/submit_eval_easy.sh

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
IMAGE="${IMAGE:-aifrontiers.azurecr.io/t-yifeili/echo-rl:latest}"
MODEL="${MODEL:-4b}"

case "$MODEL" in
    4b) DEFAULT_CONFIG="echo_configs/qwen35_4b_web_agent_easy_eval.yaml" ;;
    9b) DEFAULT_CONFIG="echo_configs/qwen35_9b_web_agent_easy_eval.yaml" ;;
    *) echo "[error] MODEL must be 4b or 9b, got '$MODEL'"; exit 1 ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
# Eval configs use num_engines=4 (tp=1) -> 4 GPUs are enough; no policy/ref.
GPUS="${GPUS:-4}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# Public WandB by default (matches the training submit scripts)
export WANDB_HOST="${WANDB_HOST:-https://api.wandb.ai}"

# Naming for the GPU monitor dashboard; the repo defaults to the 'cua' workstream.
export PRIORITY="${PRIORITY:-medium}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-medium}"
export PROJECT_NAME="${PROJECT_NAME:-cua}"

echo "[submit_eval_easy] MODEL=$MODEL CONFIG=$CONFIG GPUS=$GPUS IMAGE=$IMAGE"

# Tiny --cmd: just exec the uploaded driver. EVAL_CONFIG (a path string, not a
# secret) selects which eval config the driver runs.
bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "$IMAGE" \
    --node 1 --gpu-per-node "$GPUS" \
    --cpu 64 --memory 512Gi --shm 64Gi \
    --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
    --secret-volume echo-rl-openai:/run/secrets/echo-rl-openai \
    --extra-env-vars "EVAL_CONFIG=${CONFIG}" \
    --follow-logs \
    --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_eval_in_pod.sh'
