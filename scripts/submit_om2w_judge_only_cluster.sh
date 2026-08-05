#!/usr/bin/env bash
# Compatibility entry point. Prefer scripts/cluster/om2w/judge_only/submit.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/cluster/om2w/judge_only/submit.sh" "$@"
