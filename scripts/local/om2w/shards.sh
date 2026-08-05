#!/usr/bin/env bash
# Run an om2w benchmark sharded across several locally served vLLM instances.
#
#   PORTS="8001 8002" CFG=eval/om2w_spb_vllm_lastobs_minimal.yaml bash scripts/local/om2w/shards.sh start
#   PORTS="8000 8001" CFG=eval/om2w_spb_vllm_lastobs.yaml bash scripts/local/om2w/shards.sh start
#   bash scripts/local/om2w/shards.sh start     # build shards + launch one run per port
#   bash scripts/local/om2w/shards.sh stop      # terminate every run in this RUN group
#   bash scripts/local/om2w/shards.sh status    # pids, progress, last log line
#   bash scripts/local/om2w/shards.sh logs g2   # tail -f one shard
#   bash scripts/local/om2w/shards.sh sessions  # release leaked Browserbase sessions only
#
# sessions also takes explicit output dirs, for runs this script did not launch
# (e.g. a single unsharded scripts/local/om2w/run.sh with its own OUT):
#
#   bash scripts/local/om2w/shards.sh sessions outputs/qwen35_4b_r1
#
# stop also releases the Browserbase sessions this run leaked: killing the
# workers only drops the local CDP client, so the remote session would otherwise
# stay RUNNING for its full 1h expiry and hold concurrency. Only session ids
# recorded by THIS run are released -- the project id is shared with colleagues,
# so a project-wide release would kill their live runs. Set RELEASE_SESSIONS=0
# to skip.
#
# To test a prompt variant, point CFG at the yaml -- either an absolute path or a
# path relative to src/miniswewebagent/config/:
#
#   CFG=eval/om2w_spb_vllm_lastobs_minimal.yaml bash scripts/local/om2w/shards.sh start
#
# Each start mints a fresh run name, "<CFG basename>_<YYYYmmdd_HHMMSS>", so every
# launch gets its own output dirs (outputs/<RUN>_<shard>) and back-to-back runs of
# the same variant never overwrite each other. stop/status/logs/sessions act on
# the most recent run for that CFG (recorded in outputs/shards/<CFG basename>.current),
# so pass the same CFG when stopping. Set RUN=<name> to pin an explicit name, or
# TIMESTAMP=<stamp> to control just the suffix.
#
# Defaults are a 4-task smoke run, 2 tasks on each of two servers:
#
#   PORTS="8002 8003"   one shard per port; shard name is g<port suffix>
#   NTASKS=4            total tasks, split evenly across the shards
#   TASK_LEVEL=easy     easy|medium|hard|all
#   WORKERS=2           parallel workers per shard
#   CFG=eval/om2w_spb_vllm_lastobs.yaml
#   RUN=<CFG basename>_<timestamp>   overrides the auto-derived output/run name
#
# Override any of them inline, e.g. a 40-task medium run:
#   NTASKS=40 TASK_LEVEL=medium WORKERS=8 \
#     bash scripts/local/om2w/shards.sh start
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VENV_ROOT="${VIRTUAL_ENV:-${HOME}/.venv}"
VENV_BIN="${VENV_BIN:-$VENV_ROOT/bin}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-${CRED_FILE:-${HOME}/cred.sh}}"
# Needed for the Browserbase release call on stop; the runner sources this too.
# shellcheck disable=SC1090
[ -f "$CREDENTIALS_FILE" ] && source "$CREDENTIALS_FILE"
CFG="${CFG:-eval/om2w_spb_vllm_lastobs.yaml}"
MODEL_NAME="${MODEL_NAME:-sft_ckpt}"
TASKS_FILE="${TASKS_FILE:-$REPO/src/miniswewebagent/run/benchmarks/om2w_260220.json}"
PORTS="${PORTS:-8002 8003}"
NTASKS="${NTASKS:-4}"
TASK_LEVEL="${TASK_LEVEL:-easy}"
WORKERS="${WORKERS:-2}"

# CFG may be an absolute path or a spec relative to the builtin config dir;
# get_config_from_spec() tries the literal path first, so both reach the runner
# unchanged. Resolve here only to fail fast and to name the run.
if [ -f "$CFG" ]; then
  CFG_FILE=$CFG
elif [ -f "$REPO/src/miniswewebagent/config/$CFG" ]; then
  CFG_FILE="$REPO/src/miniswewebagent/config/$CFG"
else
  echo "config not found: $CFG" >&2
  echo "  expected an existing file, or a path under $REPO/src/miniswewebagent/config/" >&2
  exit 1
fi

# One output dir per run, timestamped, so repeated runs of the same config never
# overwrite each other. start records the name it minted; the other subcommands
# resolve it back.
RUN_BASE="$(basename "$CFG_FILE" .yaml)"
POINTER_FILE="$REPO/outputs/shards/${RUN_BASE}.current"
RUN_EXPLICIT="${RUN:-}"
TS_GLOB='[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]'

cd "$REPO"

shard_names() {
  local port
  for port in $PORTS; do echo "g${port: -1}"; done
}

# Campaigns live at outputs/runs/<RUN>/<shard>, one directory per run group, so the
# top level stays one entry per campaign instead of one per shard. The flat
# outputs/<RUN>_<shard> form predates that layout and is still resolved for runs
# started before the move.
RUNS_ROOT="$REPO/outputs/runs"

shard_out_dir() {
  local run="$1" name="$2"
  if [ ! -d "$RUNS_ROOT/$run/$name" ] && [ -d "$REPO/outputs/${run}_${name}" ]; then
    echo "$REPO/outputs/${run}_${name}"
  else
    echo "$RUNS_ROOT/$run/$name"
  fi
}

out_dir() { shard_out_dir "$RUN" "$1"; }

# Newest run group for this config, by directory mtime. The timestamp is matched
# digit by digit so a config whose name prefixes another's (e.g. lastobs vs
# lastobs_think) never adopts the other's runs; a legacy untimestamped dir from
# before this scheme is accepted as a last resort.
latest_run() {
  local dir name
  for dir in $(ls -dt "$RUNS_ROOT"/"${RUN_BASE}"_${TS_GLOB} \
                      "$REPO"/outputs/"${RUN_BASE}"_${TS_GLOB}_g? \
                      "$REPO"/outputs/"${RUN_BASE}"_g? 2>/dev/null); do
    name=$(basename "$dir")
    echo "${name%_g?}"
    return 0
  done
  return 1
}

resolve_run_for_start() {
  PREV_RUN="$(cat "$POINTER_FILE" 2>/dev/null || latest_run || true)"
  RUN="${RUN_EXPLICIT:-${RUN_BASE}_${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}}"
}

resolve_run_existing() {
  RUN="${RUN_EXPLICIT:-$(cat "$POINTER_FILE" 2>/dev/null || latest_run || true)}"
  if [ -z "$RUN" ]; then
    echo "no run found for config $RUN_BASE (looked in $POINTER_FILE and outputs/)." >&2
    echo "  Start one first, or set RUN=<name> to name it explicitly." >&2
    exit 1
  fi
}

# A run is live if its pid file names a process group that still has members.
# (`ps -p -<pgid>` is not valid, so scan every pgid instead.)
group_alive() {
  local pid=${1:-}
  [ -n "$pid" ] || return 1
  ps -eo pgid= | awk -v p="$pid" '$1==p {found=1} END {exit !found}'
}

cmd_start() {
  local name pid candidate candidates
  # Fail fast on a port with no server behind it. Without this, every task on
  # that shard dies with an httpx ConnectError on its first model.query, which
  # looks like a config bug rather than "you named a port that isn't serving".
  local port code
  for port in $PORTS; do
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/v1/models" 2>/dev/null)
    if [ "$code" != "200" ]; then
      echo "no vLLM server answering on 127.0.0.1:$port (/v1/models -> ${code:-no response})." >&2
      echo "  Serve it first, or set PORTS to the ports you are actually serving." >&2
      exit 1
    fi
  done
  # A fresh timestamp can never collide with the previous run's dirs, so check
  # that run explicitly -- otherwise a second start would quietly double up on
  # the same GPUs and Browserbase concurrency.
  candidates="$RUN"
  if [ -n "${PREV_RUN:-}" ] && [ "$PREV_RUN" != "$RUN" ]; then
    candidates="$candidates $PREV_RUN"
  fi
  for candidate in $candidates; do
    for name in $(shard_names); do
      pid=$(cat "$(shard_out_dir "$candidate" "$name")/run.pid" 2>/dev/null)
      if group_alive "${pid:-}"; then
        echo "RUN=$candidate shard $name is already running (pgid $pid). Run 'stop' first." >&2
        exit 1
      fi
    done
  done

  echo "RUN=$RUN"
  mkdir -p "$REPO/outputs/shards"
  RUN="$RUN" NTASKS="$NTASKS" TASK_LEVEL="$TASK_LEVEL" \
  SHARDS="$(shard_names | tr '\n' ' ')" TASKS_FILE="$TASKS_FILE" \
  "$VENV_BIN/python" - <<'PY' || exit 1
import json, os, pathlib, sys

run = os.environ["RUN"]
level = os.environ["TASK_LEVEL"]
total = int(os.environ["NTASKS"])
shards = os.environ["SHARDS"].split()

tasks = json.load(open(os.environ["TASKS_FILE"]))
if level != "all":
    tasks = [t for t in tasks if t.get("level") == level]
if len(tasks) < total:
    sys.exit(f"only {len(tasks)} '{level}' tasks available, need {total}")
tasks = tasks[:total]

out = pathlib.Path("outputs/shards")
# Round-robin so an uneven split spreads the remainder instead of piling it on
# the last shard.
for i, name in enumerate(shards):
    chunk = tasks[i::len(shards)]
    path = out / f"{run}_{name}.json"
    json.dump(chunk, open(path, "w"), indent=2)
    print(f"  {path}: {len(chunk)} task(s)")
PY

  # Record the run so stop/status/logs/sessions can find these dirs later,
  # since they cannot re-derive the timestamp.
  echo "$RUN" >"$POINTER_FILE"

  set -m   # give each background job its own process group, so stop can
           # signal the runner, its workers, browser_session and playwright
           # in a single kill.
  local port out
  for port in $PORTS; do
    name="g${port: -1}"
    out=$(out_dir "$name")
    rm -rf "$out"; mkdir -p "$out"
    nohup env \
      CFG="$CFG_FILE" \
      ENDPOINT="http://127.0.0.1:$port/v1" \
      MODEL_NAME="$MODEL_NAME" \
      TASKS_FILE="$REPO/outputs/shards/${RUN}_${name}.json" \
      WORKERS="$WORKERS" \
      OUT="$out" \
      bash "$REPO/scripts/local/om2w/run.sh" >"$out/launch.log" 2>&1 &
    echo $! >"$out/run.pid"
    echo "  $name: pgid=$! port=$port out=$out"
  done
  echo
  echo "Monitor: bash scripts/local/om2w/shards.sh status"
  echo "Stop:    bash scripts/local/om2w/shards.sh stop"
}

cmd_stop() {
  local name pid f
  for name in $(shard_names); do
    f="$(out_dir "$name")/run.pid"
    if [ ! -s "$f" ]; then echo "  $name: no pid file"; continue; fi
    pid=$(cat "$f")
    if kill -TERM "-$pid" 2>/dev/null; then
      echo "  $name: TERM -> process group $pid"
    else
      echo "  $name: group $pid already gone"
    fi
  done
  sleep 5
  for name in $(shard_names); do
    pid=$(cat "$(out_dir "$name")/run.pid" 2>/dev/null) || continue
    [ -n "$pid" ] || continue
    kill -KILL "-$pid" 2>/dev/null && echo "  $name: KILL -> stragglers in $pid"
  done
  # Bracket keeps the pattern from matching this script's own command line.
  echo "  remaining ${RUN} processes: $(ps -eo cmd | grep -c "${RUN}[_]g" || true)"
  release_sessions
}

# Killing the workers only drops the local CDP client; the Browserbase session
# stays RUNNING remotely until its 1h expiry, holding concurrency the whole
# time. Release the ones this run opened and never closed.
#
# Scoped deliberately: the project id is shared, so only session ids recorded by
# THIS run's browser-sessions.jsonl are touched. Never release project-wide.
#
# Takes the roots to scan as arguments; with none, defaults to this RUN's shard
# dirs. Each root is scanned exactly one level down (<root>/<task>/), which is
# where the workspace of a single task lives -- passing outputs/ itself finds
# nothing rather than silently releasing every run on the box.
release_sessions() {
  if [ "${RELEASE_SESSIONS:-1}" = "0" ]; then
    echo "  browserbase: release skipped (RELEASE_SESSIONS=0)"
    return 0
  fi
  local dirs=("$@")
  local name
  if [ ${#dirs[@]} -eq 0 ]; then
    for name in $(shard_names); do dirs+=("$(out_dir "$name")"); done
  fi
  RUN="${RUN:-}" "$VENV_BIN/python" - "${dirs[@]}" <<'PY'
import json, os, pathlib, sys, urllib.request, urllib.error

api_key = os.environ.get("BROWSERBASE_API_KEY")
if not api_key:
    print("  browserbase: BROWSERBASE_API_KEY unset, skipping release")
    raise SystemExit(0)

# created-but-not-closed == leaked by the kill
open_ids: dict[str, str] = {}
for root in sys.argv[1:]:
    for f in pathlib.Path(root).glob("*/browser-sessions.jsonl"):
        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("session_id")
            if not sid or not rec.get("owned", True):
                continue
            if rec.get("event") == "created":
                open_ids[sid] = rec.get("project_id") or ""
            elif rec.get("event") == "closed":
                open_ids.pop(sid, None)

if not open_ids:
    print("  browserbase: no leaked sessions to release")
    raise SystemExit(0)

project_id = os.environ.get("BROWSERBASE_PROJECT_ID", "")
released = failed = 0
for sid, pid in open_ids.items():
    body = json.dumps({"projectId": pid or project_id, "status": "REQUEST_RELEASE"}).encode()
    req = urllib.request.Request(
        f"https://api.browserbase.com/v1/sessions/{sid}",
        data=body,
        headers={"x-bb-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            released += 1
    except urllib.error.HTTPError as exc:
        # 4xx usually means it already expired or was released.
        if exc.code in (400, 404, 409):
            released += 1
        else:
            failed += 1
            print(f"  browserbase: release failed {sid} HTTP {exc.code}")
    except Exception as exc:
        failed += 1
        print(f"  browserbase: release failed {sid} {exc}")

print(f"  browserbase: released {released}/{len(open_ids)} leaked session(s)"
      + (f", {failed} failed" if failed else ""))
PY
}

cmd_status() {
  local name out pid dirs scored state
  echo "RUN=$RUN"
  printf '%-6s %-10s %-8s %-7s %s\n' SHARD STATE TASKS SCORED LAST
  for name in $(shard_names); do
    out=$(out_dir "$name")
    pid=$(cat "$out/run.pid" 2>/dev/null)
    if group_alive "${pid:-}"; then state="running"; else state="stopped"; fi
    dirs=$(find "$out" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    scored=$(find "$out" -name 'WebJudge_Online_Mind2Web_eval-3.json' 2>/dev/null | wc -l)
    printf '%-6s %-10s %-8s %-7s %s\n' \
      "$name" "$state" "$dirs" "$scored" "$(tail -n 1 "$out/run.log" 2>/dev/null)"
  done
}

cmd_logs() {
  local name=${1:-$(shard_names | head -1)}
  tail -f "$(out_dir "$name")/run.log"
}

case "${1:-}" in
  start)    resolve_run_for_start; cmd_start ;;
  stop)     resolve_run_existing; cmd_stop ;;
  status)   resolve_run_existing; cmd_status ;;
  logs)     resolve_run_existing; cmd_logs "${2:-}" ;;
  sessions)
    shift
    # Explicit dirs skip run resolution entirely, so this works for runs that
    # were never started through this script.
    if [ $# -gt 0 ]; then release_sessions "$@"; else resolve_run_existing; release_sessions; fi
    ;;
  *)        sed -n '2,45p' "$0"; exit 1 ;;
esac
