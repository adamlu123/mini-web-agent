#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="/home/luyadong/sandbox/mini-web-agent"

PORT="${PORT:-8787}"
HTTP_SESSION="${HTTP_SESSION:-review_viewer_http}"
TUNNEL_SESSION="${TUNNEL_SESSION:-review_viewer_cloudflare}"
TOOLS_DIR="${TOOLS_DIR:-$ROOT_DIR/.tools}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$TOOLS_DIR/cloudflared}"
PYTHON_BIN="${PYTHON_BIN:-/home/luyadong/.venv/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/outputs/sandbox}"
JUDGE_ROOT="${JUDGE_ROOT:-$REPO_ROOT/outputs/sandbox}"
HTTP_LOG="$LOG_DIR/http_server.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"
PUBLIC_URL_FILE="$LOG_DIR/cloudflare_url.txt"

mkdir -p "$TOOLS_DIR" "$LOG_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

ensure_cloudflared() {
  if [[ -x "$CLOUDFLARED_BIN" ]]; then
    return
  fi

  require_cmd curl
  local tmp_bin
  tmp_bin="$(mktemp "$TOOLS_DIR/cloudflared.XXXXXX")"
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$tmp_bin"
  chmod +x "$tmp_bin"
  mv "$tmp_bin" "$CLOUDFLARED_BIN"
}

session_exists() {
  local session_name="$1"
  screen -list | grep -q "[.]${session_name}[[:space:]]"
}

wait_for_local_server() {
  local attempt
  for attempt in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/index.html" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for local review viewer server on port ${PORT}" >&2
  exit 1
}

wait_for_public_url() {
  local attempt
  local url
  for attempt in $(seq 1 60); do
    if [[ -f "$TUNNEL_LOG" ]]; then
      url="$(grep -Eo 'https://[-[:alnum:]]+\.trycloudflare\.com' "$TUNNEL_LOG" | tail -n 1 || true)"
      if [[ -n "$url" ]]; then
        printf '%s\n' "$url" >"$PUBLIC_URL_FILE"
        printf '%s\n' "$url"
        return 0
      fi
    fi
    sleep 1
  done
  echo "Timed out waiting for Cloudflare public URL. Check $TUNNEL_LOG" >&2
  exit 1
}

start_http_screen() {
  if session_exists "$HTTP_SESSION"; then
    echo "HTTP screen session already exists: $HTTP_SESSION" >&2
    exit 1
  fi

  : >"$HTTP_LOG"
  screen -L -Logfile "$HTTP_LOG" -dmS "$HTTP_SESSION" bash -lc \
    "cd $(printf '%q' "$REPO_ROOT") && export PYTHONPATH=$(printf '%q' "$REPO_ROOT/src")\${PYTHONPATH:+:\$PYTHONPATH} && exec $(printf '%q' "$PYTHON_BIN") -m miniswewebagent.run.utilities.review_viewer --runs-root $(printf '%q' "$RUNS_ROOT") --judge-root $(printf '%q' "$JUDGE_ROOT") --host 127.0.0.1 --port $(printf '%q' "$PORT")"
}

start_tunnel_screen() {
  if session_exists "$TUNNEL_SESSION"; then
    echo "Tunnel screen session already exists: $TUNNEL_SESSION" >&2
    exit 1
  fi

  : >"$TUNNEL_LOG"
  screen -L -Logfile "$TUNNEL_LOG" -dmS "$TUNNEL_SESSION" bash -lc \
    "exec $(printf '%q' "$CLOUDFLARED_BIN") tunnel --no-autoupdate --url $(printf '%q' "http://127.0.0.1:${PORT}")"
}

main() {
  require_cmd screen
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing python executable: $PYTHON_BIN" >&2
    exit 1
  fi
  if [[ ! -d "$RUNS_ROOT" ]]; then
    echo "Runs root does not exist: $RUNS_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$JUDGE_ROOT" ]]; then
    echo "Judge root does not exist: $JUDGE_ROOT" >&2
    exit 1
  fi
  ensure_cloudflared

  start_http_screen
  wait_for_local_server
  start_tunnel_screen

  local public_url
  public_url="$(wait_for_public_url)"

  cat <<EOF
Local review viewer is running in screen session: $HTTP_SESSION
Cloudflare tunnel is running in screen session: $TUNNEL_SESSION
Local URL: http://127.0.0.1:${PORT}/index.html
Public URL: $public_url
Runs root: $RUNS_ROOT
Judge root: $JUDGE_ROOT
HTTP log: $HTTP_LOG
Tunnel log: $TUNNEL_LOG
Attach commands:
  screen -r $HTTP_SESSION
  screen -r $TUNNEL_SESSION
EOF
}

main "$@"
