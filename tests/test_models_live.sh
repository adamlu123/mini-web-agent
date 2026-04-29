#!/usr/bin/env bash
# Run the live model integration tests against the real OpenAI / Anthropic APIs.
# Sources ~/cred.sh (or $CRED_FILE) for OPENAI_API_KEY / ANTHROPIC_API_KEY.

set -euo pipefail

CRED_FILE="${CRED_FILE:-$HOME/cred.sh}"
if [[ -f "$CRED_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CRED_FILE"
else
    echo "warn: credentials file '$CRED_FILE' not found; relying on existing env" >&2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure this repo's src/ is found before any other webwright install
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -m pytest tests/test_models_live.py -s "$@"
