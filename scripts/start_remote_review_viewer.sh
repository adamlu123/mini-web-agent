#!/usr/bin/env bash
set -euo pipefail
# command
# cd /home/luyadong/sandbox/mini-web-agent
# ./scripts/start_remote_review_viewer.sh 8010
# ssh -N -L 8010:127.0.0.1:8010 <user>@<server>

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
if [[ "$BIND_HOST" == "127.0.0.1" || "$BIND_HOST" == "localhost" ]]; then
  echo "Use SSH tunnel from local machine:"
  echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} <user>@<server>"
  echo "Then open: http://127.0.0.1:${PORT}"
  echo "Or create a public share link from this machine:"
  echo "  bash ./scripts/start_public_tunnel.sh ${PORT}"
else
  echo "Open directly: http://<server-ip>:${PORT}"
fi

echo ""
PYTHONPATH=src "$PYTHON_BIN" -m miniswewebagent.run.utilities.review_viewer \
  --runs-root "$RUNS_ROOT" \
  --judge-root "$JUDGE_ROOT" \
  --host "$BIND_HOST" \
  --port "$PORT"
