#!/usr/bin/env bash
# Image smoke test for echo-rl on Lambda.
# Submits a tiny 1-node 1-GPU job that:
#   1. uploads mini-web-agent + SkyRL into PVC
#   2. pip install -e's both editables in the container
#   3. imports echo_rl + skyrl, launches a headless Chromium via playwright,
#      navigates to about:blank, exits.
#
# Use this after `bash docker/build.sh` to verify the image works end-to-end
# before submitting a real training run.
#
# Usage:
#   bash docker/submit_smoke.sh
#   TAG=v2 bash docker/submit_smoke.sh

set -euo pipefail

REGISTRY="${REGISTRY:-aifrontiers}"
IMAGE="${IMAGE:-t-yifeili/echo-rl}"
TAG="${TAG:-latest}"
SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"

MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    if [[ ! -d "$d" ]]; then
        echo "[error] Missing repo: $d"
        exit 1
    fi
done

export PATH="$HOME/.krew/bin:$PATH"

# Job priority. See submit_real_train_batch.sh for explanation.
export PRIORITY="${PRIORITY:-medium}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-medium}"

# 1 GPU is enough to verify everything resolves; bump --gpu-per-node to 8 for
# a real run once this passes.
bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --image "${REGISTRY}.azurecr.io/${IMAGE}:${TAG}" \
    --node 1 \
    --gpu-per-node 1 \
    --cpu 8 \
    --memory 32Gi \
    --shm 8Gi \
    --follow-logs \
    --cmd 'set -e
echo "[smoke] hello from $JOB_NAME on $(hostname)"
echo "[smoke] $(date) — pwd=$(pwd)"

# submit_job.sh sets workingDir to the primary upload (mini-web-agent).
# SkyRL was uploaded alongside; resolve its path on PVC.
SKYRL_ON_PVC="${PVC_MOUNT}/${USER_ALIAS}/runs/${JOB_NAME}/SkyRL"
MWA_ON_PVC="${PVC_MOUNT}/${USER_ALIAS}/runs/${JOB_NAME}/mini-web-agent"
echo "[smoke] MWA_ON_PVC=$MWA_ON_PVC"
echo "[smoke] SKYRL_ON_PVC=$SKYRL_ON_PVC"
ls "$SKYRL_ON_PVC" | head

echo "[smoke] === installing editables ==="
pip install --no-deps -e "$SKYRL_ON_PVC"
pip install --no-deps -e "$SKYRL_ON_PVC/skyrl-gym"
pip install --no-deps -e "$SKYRL_ON_PVC/skyrl-agent"
pip install --no-deps -e "$MWA_ON_PVC"

echo "[smoke] === import checks ==="
python -c "
import importlib.metadata as md
import echo_rl, skyrl, skyrl_gym, skyrl_agent
import playwright, torch, vllm
print(\"echo_rl       \", echo_rl.__file__)
print(\"skyrl         \", skyrl.__file__)
print(\"skyrl_gym     \", skyrl_gym.__file__)
print(\"skyrl_agent   \", skyrl_agent.__file__)
print(\"playwright    \", md.version(\"playwright\"))
print(\"torch         \", torch.__version__, \"cuda?\", torch.cuda.is_available(), \"count\", torch.cuda.device_count())
print(\"vllm          \", vllm.__version__)
"

echo "[smoke] === playwright launch ==="
python <<PY
import asyncio
from playwright.async_api import async_playwright

async def go():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        await page.goto("about:blank")
        print("[smoke] page title:", repr(await page.title()))
        await browser.close()

asyncio.run(go())
PY

echo "[smoke] done — exiting"'
