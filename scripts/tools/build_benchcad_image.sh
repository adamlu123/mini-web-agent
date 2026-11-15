#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
benchcad_repo="${1:-$repo_root/cache/BenchCAD-main}"
image="${2:-cuawright-benchcad:52087ef}"

test "$(git -C "$benchcad_repo" rev-parse HEAD)" = \
  "52087ef4b08811c21c09b8b16d0ae36c70844865"
test -z "$(git -C "$benchcad_repo" status --porcelain)"

docker buildx build --load \
  --build-context "benchcad=$benchcad_repo" \
  --build-arg "BENCHCAD_COMMIT=52087ef4b08811c21c09b8b16d0ae36c70844865" \
  --file "$repo_root/src/cuawright/environments/benchcad/container/Dockerfile" \
  --tag "$image" \
  "$repo_root"
