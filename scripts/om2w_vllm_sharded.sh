#!/usr/bin/env bash
# Compatibility entry point. Prefer scripts/local/om2w/shards.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/local/om2w/shards.sh" "$@"
