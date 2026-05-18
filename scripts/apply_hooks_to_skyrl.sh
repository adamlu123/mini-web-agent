#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "" ]]; then
  echo "usage: $0 /path/to/SkyRL" >&2
  exit 2
fi

SKYRL_DIR=$(cd "$1" && pwd)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXPECTED_BASE="43aab09782953cc7cfc93bda52b1635d717ce446"

if ! git -C "$SKYRL_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: $SKYRL_DIR is not a git checkout" >&2
  exit 1
fi
if [[ -n $(git -C "$SKYRL_DIR" status --short) ]]; then
  echo "error: SkyRL checkout has uncommitted changes; apply to a clean checkout" >&2
  exit 1
fi
current=$(git -C "$SKYRL_DIR" rev-parse HEAD)
if [[ "$current" != "$EXPECTED_BASE" ]]; then
  echo "warning: SkyRL checkout is at $current, expected $EXPECTED_BASE" >&2
fi

git -C "$SKYRL_DIR" apply --check "$REPO_ROOT/patches/skyrl_minimal_hooks.patch"
git -C "$SKYRL_DIR" apply "$REPO_ROOT/patches/skyrl_minimal_hooks.patch"
cp -a "$REPO_ROOT/echo_rl" "$SKYRL_DIR/"
mkdir -p "$SKYRL_DIR/echo_configs"
cp -a "$REPO_ROOT/configs/." "$SKYRL_DIR/echo_configs/"
cp "$REPO_ROOT/scripts/run_echo_terminal_agent.sh" "$SKYRL_DIR/run_echo_terminal_agent.sh"
chmod +x "$SKYRL_DIR/run_echo_terminal_agent.sh"

echo "Applied ECHO minimal hooks and copied readable ECHO files into $SKYRL_DIR"
