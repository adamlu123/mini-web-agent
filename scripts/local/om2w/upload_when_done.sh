#!/usr/bin/env bash
# Wait for the om2w batch to finish, then upload its output directory to blob.
#
#   OUT=outputs/default/<batch> DEST=https://<account>.blob.core.windows.net/<container>/<path> \
#     bash scripts/local/om2w/upload_when_done.sh
#
# "Finished" means no om2w worker processes remain. Tasks that hang without ever
# writing result.json would block that forever, so DEADLINE_MIN caps the wait:
# once it expires the remaining workers are killed and whatever completed is
# uploaded.
set -uo pipefail

VENV_ROOT="${VIRTUAL_ENV:-${HOME}/.venv}"
VENV_BIN="${VENV_BIN:-$VENV_ROOT/bin}"

OUT="${OUT:?set OUT to the om2w batch output directory}"
DEST="${DEST:?set DEST to the destination blob URL}"
[[ -d "$OUT" ]] || { echo "output directory not found: $OUT" >&2; exit 1; }
[[ -x "$VENV_BIN/python" ]] || {
    echo "Python executable not found: $VENV_BIN/python" >&2
    exit 1
}
LOG="${LOG:-$OUT/_resume/upload.log}"
mkdir -p "$(dirname "$LOG")"
POLL="${POLL:-60}"
DEADLINE_MIN=${DEADLINE_MIN:-30}   # upload no later than this many minutes from start

export AZCOPY_AUTO_LOGIN_TYPE=AZCLI

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

deadline_ts=$(( $(date +%s) + DEADLINE_MIN * 60 ))
log "watcher started; upload deadline $(date -d "@$deadline_ts" '+%F %T') (${DEADLINE_MIN} min)"

forced=0
while true; do
    workers=$(pgrep -f 'benchmarks[.]om2w' | wc -l)
    done_count=$(ls "$OUT"/*/result.json 2>/dev/null | wc -l)

    if [ "$workers" -eq 0 ]; then
        log "no workers remain; batch finished with $done_count completed tasks"
        break
    fi

    if [ "$(date +%s)" -ge "$deadline_ts" ]; then
        log "DEADLINE reached with $done_count completed and $workers workers alive"
        forced=1
        break
    fi
    sleep "$POLL"
done

if [ "$forced" -eq 1 ]; then
    # Stop the run before copying so azcopy never reads a half-written file.
    log "stopping remaining workers for a consistent snapshot"
    "$VENV_BIN/python" - <<'PY' 2>&1 | tee -a "$LOG"
import os, signal, subprocess, time
pids = [int(p) for p in subprocess.run(
    ["pgrep", "-f", "benchmarks[.]om2w"], capture_output=True, text=True).stdout.split()]
print(f"killing {len(pids)} processes")
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
time.sleep(6)
left = subprocess.run(["pgrep", "-f", "benchmarks[.]om2w"],
                      capture_output=True, text=True).stdout.split()
print(f"remaining: {len(left)}")
PY
    sleep 5
    log "final completed count: $(ls "$OUT"/*/result.json 2>/dev/null | wc -l)"
fi

log "starting upload of $OUT -> $DEST"
du -sh "$OUT" | tee -a "$LOG"

azcopy copy "$OUT/" "$DEST" --recursive >>"$LOG" 2>&1
status=$?

if [ $status -eq 0 ]; then
    log "UPLOAD COMPLETE (exit 0)"
else
    log "UPLOAD FAILED (exit $status) - see $LOG"
fi
exit $status
