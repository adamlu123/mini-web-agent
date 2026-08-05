#!/usr/bin/env bash
# Shared helpers for OM2W cluster runtime scripts.

mwa_copy_staged_repo() {
    local upload_repo="$1"
    local local_repo="$2"
    local resolved_local_repo

    resolved_local_repo="$(
        python - "$local_repo" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).resolve())
PY
    )"
    case "$(basename "$resolved_local_repo")" in
        mini-web-agent-*) ;;
        *)
            echo "[cluster-runtime][error] refusing unsafe local staging path: $resolved_local_repo" >&2
            return 1
            ;;
    esac

    rm -rf -- "$resolved_local_repo"
    mkdir -p "$resolved_local_repo"
    cp -R --no-preserve=mode,ownership,timestamps "$upload_repo/." "$resolved_local_repo/"
}

mwa_verify_sha256() {
    local log_prefix="$1"
    local file_path="$2"
    local expected="$3"
    local actual

    [[ -z "$expected" ]] && return 0
    actual="$(sha256sum "$file_path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "[$log_prefix][error] checksum mismatch for $file_path" >&2
        echo "[$log_prefix][error] expected=$expected actual=$actual" >&2
        return 1
    }
}

mwa_stop_process() {
    local pid="${1:-}"
    if [[ -n "$pid" ]]; then
        kill "$pid" >/dev/null 2>&1 || true
    fi
}

mwa_wait_for_vllm() {
    local log_prefix="$1"
    local pid="$2"
    local port="$3"
    local timeout="$4"

    python - "$log_prefix" "$pid" "$port" "$timeout" <<'PY'
import sys
import time
import urllib.request
from pathlib import Path

log_prefix = sys.argv[1]
pid, port, timeout = map(int, sys.argv[2:])
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print(f"[{log_prefix}] vLLM ready: {url}", flush=True)
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        if not Path(f"/proc/{pid}/stat").exists():
            raise SystemExit(
                f"[{log_prefix}][error] vLLM exited before readiness"
            )
        time.sleep(5)
raise SystemExit(f"[{log_prefix}][error] vLLM readiness timed out: {url}")
PY
}
