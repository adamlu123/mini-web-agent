# Image for echo-rl web-agent runs on the Lambda (bonete61) cluster.
#
# Strategy: bake every heavy dep (273-pkg pip freeze) into the image so pods
# come up in ~1-2 min. Editable installs (echo_rl, skyrl, skyrl-gym,
# skyrl-agent) are NOT baked — they're uploaded per-job via submit_job.sh
# and pip-install -e'd at runtime so code edits don't require a rebuild.
#
# Build (cloud, no local Docker needed):
#   az acr build --registry aifrontiers --image t-yifeili/echo-rl:v1 \
#       --file docker/Dockerfile .

FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Bootstrap apt + the few extras playwright doesn't pull in itself. The
# Chromium system libs are installed below via `playwright install --with-deps`
# which knows the correct package names for the host Ubuntu version (the base
# is Ubuntu 24.04 "noble" where many libs got renamed to t64 variants).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        xdg-utils \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Install the full frozen dep set in one layer. --no-deps is intentional:
# we already have a fully-resolved freeze, and letting pip re-resolve here
# tends to thrash on the cuda-* / nvidia-* metapackages.
#
# docker/wheels/ ships prebuilt linux_x86_64 wheels for causal-conv1d and
# flash_attn (compiled locally against torch 2.10 + glibc 2.39). pip's
# --find-links resolves the entries in requirements.txt against these local
# wheels before falling back to PyPI, which skips a 30-60 min CUDA template
# compile on ACR's 2-core build agent.
COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/wheels /tmp/wheels
RUN pip install --no-cache-dir --no-deps \
        --find-links /tmp/wheels \
        -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels

# `playwright install --with-deps chromium` runs apt-get under the hood with
# the correct package names per Ubuntu version, then drops Chromium into the
# cache below. Need apt cache repopulated since we cleaned it above.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN apt-get update \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Sanity check: imports must succeed in the built image. playwright module
# doesn't expose __version__; use importlib.metadata for version lookups.
RUN python -c "import importlib.metadata as md; \
import torch, vllm, playwright, transformers, accelerate, ray; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('vllm', vllm.__version__); \
print('playwright', md.version('playwright'))"
