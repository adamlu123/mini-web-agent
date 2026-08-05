#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8010}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
RUNS_ROOT="${RUNS_ROOT:-outputs/sandbox}"
JUDGE_ROOT="${JUDGE_ROOT:-om2w_judge}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

echo "Serving review viewer from: $REPO_ROOT"
echo "Bind host: $BIND_HOST"
echo "Port: $PORT"
echo "RUNS_ROOT: $RUNS_ROOT"
echo "JUDGE_ROOT: $JUDGE_ROOT"
echo ""
echo "Open: http://127.0.0.1:${PORT}"
echo ""

PYTHONPATH=src "$PYTHON_BIN" -m miniswewebagent.run.utilities.review_viewer \
  --runs-root "$RUNS_ROOT" \
  --judge-root "$JUDGE_ROOT" \
  --host "$BIND_HOST" \
  --port "$PORT"
