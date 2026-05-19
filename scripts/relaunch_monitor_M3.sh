#!/usr/bin/env bash
# Monitor /outputs/default/0426/0426_self_judge_M3 every 30 min and re-launch
# tasks that finished with LimitsExceeded or HTTPStatusError. Each retry's
# result is written to the same relaunch dir, overwriting the prior attempt.
# Loop continues until no LimitsExceeded/HTTPStatusError remain in either the
# source dir (for tasks never relaunched) or the relaunch dir.
set -uo pipefail

SOURCE_DIR=/home/luyadong/sandbox/mini-web-agent/outputs/default/0426/0426_self_judge_M3
RELAUNCH_DIR=/home/luyadong/sandbox/mini-web-agent/outputs/default/0426/0426_self_judge_M3_reluanch
REPO=/home/luyadong/sandbox/mini-web-agent
INTERVAL=1800   # 30 minutes
LOG="$RELAUNCH_DIR/monitor.log"

mkdir -p "$RELAUNCH_DIR"
touch "$LOG"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

source /home/luyadong/cred.sh

iteration=0
while true; do
    iteration=$((iteration+1))
    log "=== iteration $iteration: scanning ==="

    # Determine current set of failing task ids:
    # - For tasks present in RELAUNCH_DIR, prefer that result (latest attempt).
    # - Otherwise fall back to SOURCE_DIR result.
    # A task is "failing" if its latest result has exit_status==LimitsExceeded
    # or HTTPStatusError (in exit_status or run_exception).
    # We only emit ids that are NOT currently being processed (i.e. monitor
    # only relaunches when no batch is running).
    failed_ids=$(python3 - "$SOURCE_DIR" "$RELAUNCH_DIR" <<'PY'
import json, glob, os, sys
src, relaunch = sys.argv[1:3]

def latest(tid):
    for d in (relaunch, src):
        p = os.path.join(d, tid, "result.json")
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return None
    return None

def is_failing(d):
    if not d: return False
    es = d.get("exit_status") or ""
    rex = str(d.get("run_exception") or "")
    return es == "LimitsExceeded" or "HTTPStatusError" in es or "HTTPStatusError" in rex

ids = set()
for p in glob.glob(os.path.join(src, "*", "result.json")):
    tid = os.path.basename(os.path.dirname(p))
    if is_failing(latest(tid)):
        ids.add(tid)
# Also catch tasks that only have a relaunch result (defensive)
for p in glob.glob(os.path.join(relaunch, "*", "result.json")):
    tid = os.path.basename(os.path.dirname(p))
    if is_failing(latest(tid)):
        ids.add(tid)
print("\n".join(sorted(ids)))
PY
)

    failed_count=$(printf "%s" "$failed_ids" | grep -c . || true)
    log "found $failed_count failing task(s)"

    if [ "$failed_count" -eq 0 ]; then
        log "no failing tasks remain — exiting monitor"
        break
    fi

    args=()
    while IFS= read -r tid; do
        [ -z "$tid" ] && continue
        args+=(--task-id "$tid")
        # remove any stale per-task dir in relaunch so the rerun starts fresh
        rm -rf "$RELAUNCH_DIR/$tid" 2>/dev/null || true
    done <<< "$failed_ids"

    run_log="$RELAUNCH_DIR/relaunch_iter${iteration}_$(date +%Y%m%d_%H%M%S).log"
    log "launching relaunch batch (iter=$iteration, n=$failed_count) -> $run_log"
    (
        cd "$REPO" && \
        /home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
            -c best_default_judge_json.yaml \
            --workers 30 \
            --output-dir "$RELAUNCH_DIR" \
            "${args[@]}"
    ) >> "$run_log" 2>&1
    rc=$?
    log "relaunch batch iter=$iteration exited rc=$rc"

    log "sleeping $INTERVAL s before next scan"
    sleep "$INTERVAL"
done

log "monitor finished cleanly"

