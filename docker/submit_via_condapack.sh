#!/usr/bin/env bash
# Submit a job using the conda-pack'd echo-rl env, NOT a custom Docker image.
# Use this when:
#   - The custom image (docker/build.sh) doesn't exist or you don't want to wait
#     for a 30-60 min image rebuild after every dep change.
#   - You changed pip deps locally: just re-run `conda pack` and re-upload tars.
#
# One-time PVC setup (do once, then ignore):
#   mkdir -p /data/t-yifeili/echo-rl-envs
#   mv /data/t-yifeili/echo-rl.tar.gz \
#      /data/t-yifeili/playwright-chromium.tar.gz \
#      /data/t-yifeili/echo-rl-envs/
#   bash /data/t-yifeili/aifsdk/clusters/lambda/submission/pvc_data_copy.sh \
#       /data/t-yifeili/echo-rl-envs t-yifeili/envs
#   # tars now live at /mnt/pvc/t-yifeili/envs/{echo-rl,playwright-chromium}.tar.gz
#
# After that, per-job:
#   bash docker/submit_via_condapack.sh

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
MINI_WEB_AGENT_DIR="${MINI_WEB_AGENT_DIR:-/data/t-yifeili/mini-web-agent}"
SKYRL_DIR="${SKYRL_DIR:-/data/t-yifeili/SkyRL}"
ENV_TAR_PATH="${ENV_TAR_PATH:-/mnt/pvc/t-yifeili/envs/echo-rl.tar.gz}"
PW_TAR_PATH="${PW_TAR_PATH:-/mnt/pvc/t-yifeili/envs/playwright-chromium.tar.gz}"

for d in "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR"; do
    [[ -d "$d" ]] || { echo "[error] missing $d"; exit 1; }
done

export PATH="$HOME/.krew/bin:$PATH"

# Job priority. See submit_real_train_batch.sh for explanation.
export PRIORITY="${PRIORITY:-high}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"

bash "$SUBMIT" \
    --upload "$MINI_WEB_AGENT_DIR" "$SKYRL_DIR" \
    --node 1 --gpu-per-node 1 \
    --cpu 16 --memory 64Gi --shm 8Gi \
    --follow-logs \
    --cmd "set -euo pipefail
ENV_TAR='${ENV_TAR_PATH}'
PW_TAR='${PW_TAR_PATH}'

echo '[smoke] === extracting conda env (~7 GB → /opt/echo-rl) ==='
mkdir -p /opt/echo-rl
time tar -xzf \"\$ENV_TAR\" -C /opt/echo-rl
source /opt/echo-rl/bin/activate
# conda-unpack rewrites any absolute prefix paths recorded at pack time so the
# env works under /opt/echo-rl regardless of where it was packed.
conda-unpack
which python && python --version

echo '[smoke] === extracting playwright browsers ==='
export PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
mkdir -p \"\$PLAYWRIGHT_BROWSERS_PATH\"
time tar -xzf \"\$PW_TAR\" -C \"\$PLAYWRIGHT_BROWSERS_PATH\"

echo '[smoke] === apt-installing playwright system libs (chromium needs ~25 libs) ==='
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
time python -m playwright install-deps chromium

echo '[smoke] === installing editables from uploaded repos ==='
SKYRL_ON_PVC=\"\$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/SkyRL\"
MWA_ON_PVC=\"\$PVC_MOUNT/\$USER_ALIAS/runs/\$JOB_NAME/mini-web-agent\"
pip install --no-deps -e \"\$SKYRL_ON_PVC\"
pip install --no-deps -e \"\$SKYRL_ON_PVC/skyrl-gym\"
pip install --no-deps -e \"\$SKYRL_ON_PVC/skyrl-agent\"
pip install --no-deps -e \"\$MWA_ON_PVC\"

echo '[smoke] === sanity checks ==='
python -c \"
import importlib.metadata as md
import torch, vllm, playwright, transformers, accelerate, ray
import echo_rl, skyrl, skyrl_gym, skyrl_agent
print('torch     ', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())
print('vllm      ', vllm.__version__)
print('playwright', md.version('playwright'))
print('echo_rl   ', echo_rl.__file__)
print('skyrl     ', skyrl.__file__)
\"

echo '[smoke] === playwright launch (chromium → about:blank) ==='
python - <<'PY'
import asyncio
from playwright.async_api import async_playwright

async def go():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = await b.new_page()
        await page.goto('about:blank')
        print('[smoke] page title:', repr(await page.title()))
        await b.close()

asyncio.run(go())
PY

echo '[smoke] done'
"
