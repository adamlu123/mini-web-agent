#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cadgenbench_repo="${1:-$repo_root/cache/cadgenbench}"
image="${2:-cuawright-cadgenbench:33304cf}"

test "$(git -C "$cadgenbench_repo" rev-parse HEAD)" = \
  "33304cf771fc5639144b1df9611e347251052cf8"
test -z "$(git -C "$cadgenbench_repo" status --porcelain)"

docker buildx build --load \
  --build-context "cadgenbench=$cadgenbench_repo" \
  --build-arg "CADGENBENCH_COMMIT=33304cf771fc5639144b1df9611e347251052cf8" \
  --build-arg "BUILD123D_MCP_VERSION=0.3.59" \
  --file "$repo_root/src/cuawright/environments/cadgenbench/container/Dockerfile" \
  --tag "$image" \
  "$repo_root"
