#!/usr/bin/env bash
# Cloud-build the echo-rl runtime image in aifrontiers ACR.
# Pushes aifrontiers.azurecr.io/t-yifeili/echo-rl:<TAG>.
#
# Pre-req: `az login` with your sc-alt account (and access to aifrontiers ACR).
#
# Usage:
#   bash docker/build.sh                # tag = current date YYYYMMDD
#   TAG=v2 bash docker/build.sh         # explicit tag
#   REGISTRY=other bash docker/build.sh # different ACR

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
REGISTRY="${REGISTRY:-aifrontiers}"
IMAGE="${IMAGE:-t-yifeili/echo-rl}"
TAG="${TAG:-$(date -u +%Y%m%d)}"

cd "$REPO_ROOT"

if ! az account show >/dev/null 2>&1; then
    echo "[error] Run 'az login' first (use your sc-alt account)."
    exit 1
fi

echo "[build] registry=${REGISTRY}  image=${IMAGE}  tag=${TAG}"
echo "[build] context=$(pwd)  dockerfile=docker/Dockerfile"
echo "[build] build context will be ~$(du -sh --apparent-size docker/requirements.txt 2>/dev/null | cut -f1) (per .dockerignore)"

# --no-logs would suppress streaming output; we want it so failures are visible.
az acr build \
    --registry "$REGISTRY" \
    --image "${IMAGE}:${TAG}" \
    --image "${IMAGE}:latest" \
    --file docker/Dockerfile \
    .

echo
echo "[done] Image pushed:"
echo "  ${REGISTRY}.azurecr.io/${IMAGE}:${TAG}"
echo "  ${REGISTRY}.azurecr.io/${IMAGE}:latest"
echo
echo "Submit a job with:"
echo "  --image ${REGISTRY}.azurecr.io/${IMAGE}:${TAG} --acr"
