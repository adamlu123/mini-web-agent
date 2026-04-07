#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8010}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/.tools"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-${TOOLS_DIR}/cloudflared}"

download_cloudflared() {
  local os arch url tmp_file
  os="$(uname -s)"
  arch="$(uname -m)"

  if [[ "$os" != "Linux" ]]; then
    echo "This helper currently supports Linux only." >&2
    exit 1
  fi

  case "$arch" in
    x86_64|amd64)
      url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
      ;;
    aarch64|arm64)
      url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
      ;;
    *)
      echo "Unsupported architecture: $arch" >&2
      exit 1
      ;;
  esac

  mkdir -p "$TOOLS_DIR"
  tmp_file="${CLOUDFLARED_BIN}.tmp"

  echo "Downloading cloudflared to ${CLOUDFLARED_BIN}"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$tmp_file"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_file" "$url"
  else
    echo "Need curl or wget to download cloudflared." >&2
    exit 1
  fi

  chmod +x "$tmp_file"
  mv "$tmp_file" "$CLOUDFLARED_BIN"
}

if ! command -v cloudflared >/dev/null 2>&1 && [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  download_cloudflared
fi

if command -v cloudflared >/dev/null 2>&1; then
  CLOUDFLARED_BIN="$(command -v cloudflared)"
fi

echo "Creating public tunnel for http://127.0.0.1:${PORT}"
echo "Keep this process running to keep the public URL alive."
exec "$CLOUDFLARED_BIN" tunnel --url "http://127.0.0.1:${PORT}"
