#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8010}"
CLOUDFLARED_VERSION="2026.3.0"
CACHE_BASE="${XDG_CACHE_HOME:-${HOME}/.cache}"
CLOUDFLARED_CACHE_DIR="${CLOUDFLARED_CACHE_DIR:-${CACHE_BASE}/mini-web-agent/cloudflared/${CLOUDFLARED_VERSION}}"
CLOUDFLARED_BIN_OVERRIDE="${CLOUDFLARED_BIN:-}"
CLOUDFLARED_BIN="${CLOUDFLARED_CACHE_DIR}/cloudflared"

download_cloudflared() {
  local os arch asset expected_sha url tmp_file actual_sha
  os="$(uname -s)"
  arch="$(uname -m)"

  if [[ "$os" != "Linux" ]]; then
    echo "This helper currently supports Linux only." >&2
    exit 1
  fi

  case "$arch" in
    x86_64|amd64)
      asset="cloudflared-linux-amd64"
      expected_sha="4a9e50e6d6d798e90fcd01933151a90bf7edd99a0a55c28ad18f2e16263a5c30"
      ;;
    aarch64|arm64)
      asset="cloudflared-linux-arm64"
      expected_sha="0755ba4cbab59980e6148367fcf53a8f3ec85a97deefd63c2420cf7850769bee"
      ;;
    *)
      echo "Unsupported architecture: $arch" >&2
      exit 1
      ;;
  esac

  url="https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/${asset}"
  mkdir -p "$(dirname "$CLOUDFLARED_BIN")"
  tmp_file="$(mktemp "${CLOUDFLARED_BIN}.tmp.XXXXXX")"

  echo "Downloading cloudflared ${CLOUDFLARED_VERSION} to ${CLOUDFLARED_BIN}"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location "$url" --output "$tmp_file"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_file" "$url"
  else
    rm -f -- "$tmp_file"
    echo "Need curl or wget to download cloudflared." >&2
    exit 1
  fi

  actual_sha="$(sha256sum "$tmp_file" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    rm -f -- "$tmp_file"
    echo "cloudflared checksum mismatch for ${asset}" >&2
    echo "expected=${expected_sha} actual=${actual_sha}" >&2
    exit 1
  fi

  chmod +x "$tmp_file"
  mv -- "$tmp_file" "$CLOUDFLARED_BIN"
}

if [[ -n "$CLOUDFLARED_BIN_OVERRIDE" ]]; then
  CLOUDFLARED_BIN="$CLOUDFLARED_BIN_OVERRIDE"
  [[ -x "$CLOUDFLARED_BIN" ]] || {
    echo "CLOUDFLARED_BIN is not executable: $CLOUDFLARED_BIN" >&2
    exit 1
  }
elif command -v cloudflared >/dev/null 2>&1; then
  CLOUDFLARED_BIN="$(command -v cloudflared)"
elif [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  download_cloudflared
fi

echo "Creating public tunnel for http://127.0.0.1:${PORT}"
echo "Keep this process running to keep the public URL alive."
exec "$CLOUDFLARED_BIN" tunnel --url "http://127.0.0.1:${PORT}"
