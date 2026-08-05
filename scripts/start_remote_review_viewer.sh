#!/usr/bin/env bash
# Compatibility entry point. Prefer scripts/review/viewer.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/review/viewer.sh" "$@"
