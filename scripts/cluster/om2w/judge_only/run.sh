#!/usr/bin/env bash
# Re-score an existing OM2W eval run on a CPU-only pod.
#
# Generation is the expensive, GPU-bound half; the judge only calls the
# gateway. When a judge pass fails (e.g. a clobbered task.json aborting task
# discovery) the trajectories on the PVC are still good, so re-run just the
# judge instead of the whole eval.
#
# Required: DATA_ROOT, JOB_NAME, RUN_ROOT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../lib/cluster_runtime.sh
source "$SCRIPT_DIR/../../../lib/cluster_runtime.sh"

: "${DATA_ROOT:?DATA_ROOT is not set}"
: "${JOB_NAME:?JOB_NAME is not set}"
: "${RUN_ROOT:?RUN_ROOT is not set}"

UPLOAD_REPO="${UPLOAD_REPO:-$DATA_ROOT/runs/$JOB_NAME/mini-web-agent}"
LOCAL_REPO="${LOCAL_REPO:-/tmp/mini-web-agent-judge-only}"
ASSET_SUBDIR="${ASSET_SUBDIR:-cluster_eval_assets}"
TRAJECTORIES_DIR="${TRAJECTORIES_DIR:-$RUN_ROOT/outputs}"
EVAL_DIR_NAME="${EVAL_DIR_NAME:-outputs_eval_judgeonly_1}"
JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
JUDGE_NUM_PROC="${JUDGE_NUM_PROC:-32}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://gateway.phyagi.net/api/responses}"
# "none" 表示直连官方 OpenAI(空 endpoint_target_uri,用 OPENAI_API_KEY)
[[ "$JUDGE_ENDPOINT" == "none" ]] && JUDGE_ENDPOINT=""
SCORE_THRESHOLD="${SCORE_THRESHOLD:-3}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"

[[ "$JUDGE_NUM_PROC" =~ ^[1-9][0-9]*$ ]] || {
    echo "[judge-only][error] JUDGE_NUM_PROC must be a positive integer: $JUDGE_NUM_PROC" >&2
    exit 2
}
[[ -d "$UPLOAD_REPO" ]] || {
    echo "[judge-only][error] staged repository is missing: $UPLOAD_REPO" >&2
    exit 1
}
[[ -d "$TRAJECTORIES_DIR" ]] || {
    echo "[judge-only][error] trajectories directory is missing: $TRAJECTORIES_DIR" >&2
    exit 1
}
[[ -f "$CREDS_FILE" ]] || {
    echo "[judge-only][error] credentials file is missing: $CREDS_FILE" >&2
    exit 1
}

mwa_copy_staged_repo "$UPLOAD_REPO" "$LOCAL_REPO"
REPO="$LOCAL_REPO"
JUDGE_SCRIPT="$REPO/scripts/eval/persistent_cli_steps.py"
[[ -f "$JUDGE_SCRIPT" ]] || {
    echo "[judge-only][error] judge script is missing: $JUDGE_SCRIPT" >&2
    exit 1
}

EVAL_OUTPUT_DIR="$RUN_ROOT/$EVAL_DIR_NAME"
LOGS_DIR="$RUN_ROOT/logs/judge_only_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EVAL_OUTPUT_DIR" "$LOGS_DIR"

source "$CREDS_FILE"
if [[ -n "${PHYAGI_API_KEY:-}" ]]; then
    export OPENAI_GATEWAY_API_KEY="${OPENAI_GATEWAY_API_KEY:-$PHYAGI_API_KEY}"
fi
if [[ -n "$JUDGE_ENDPOINT" ]]; then
    : "${OPENAI_GATEWAY_API_KEY:?OPENAI_GATEWAY_API_KEY is not set by the credentials file}"
    JUDGE_API_KEY="$OPENAI_GATEWAY_API_KEY"
else
    : "${OPENAI_API_KEY:?direct-OpenAI judge (JUDGE_ENDPOINT=none) needs OPENAI_API_KEY in the credentials file}"
    JUDGE_API_KEY="$OPENAI_API_KEY"
fi

export PYTHONPATH="$REPO/agent_runtime${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

echo "[judge-only] installing judge dependencies"
# Only what the judge imports: om2w_judge.utils needs backoff/httpx/openai and
# the WebJudge method needs Pillow. No browser or vLLM stack on a CPU pod.
python -m pip install --no-deps -e "$REPO"
python -m pip install --no-deps backoff httpx openai pillow
python -c 'import backoff, httpx, openai, PIL, miniswewebagent'

TASK_DIR_COUNT="$(find "$TRAJECTORIES_DIR" -mindepth 2 -maxdepth 2 -name task.json | wc -l)"
echo "[judge-only] run_root=$RUN_ROOT"
echo "[judge-only] trajectories=$TRAJECTORIES_DIR task_dirs=$TASK_DIR_COUNT"
echo "[judge-only] model=$JUDGE_MODEL num_proc=$JUDGE_NUM_PROC endpoint=$JUDGE_ENDPOINT"
echo "[judge-only] eval_output=$EVAL_OUTPUT_DIR"

cd "$REPO"
set +e
python "$JUDGE_SCRIPT" \
    --model "$JUDGE_MODEL" \
    --trajectories_dir "$TRAJECTORIES_DIR" \
    --api_key "$JUDGE_API_KEY" \
    --output_path "$EVAL_OUTPUT_DIR" \
    --num_worker "$JUDGE_NUM_PROC" \
    --score_threshold "$SCORE_THRESHOLD" \
    --expected_tasks 0 \
    --endpoint_target_uri "$JUDGE_ENDPOINT" 2>&1 | tee "$LOGS_DIR/judge.log"
JUDGE_RC=${PIPESTATUS[0]}
set -e

if [[ "$JUDGE_RC" != "0" ]]; then
    echo "[judge-only][error] judge exited rc=$JUDGE_RC" >&2
    exit "$JUDGE_RC"
fi

echo "[judge-only] normalizing per-task score files"
python - "$EVAL_OUTPUT_DIR" "$JUDGE_MODEL" <<'PY'
import sys
from pathlib import Path

from miniswewebagent.utils.om2w_eval import (
    JudgeInterface,
    judge_result_file_path,
    normalize_online_mind2web_judge_results,
)

eval_dir, judge_model = sys.argv[1:]
result_file = judge_result_file_path(
    Path(eval_dir),
    judge_model,
    judge_interface=JudgeInterface.PERSISTENT_CLI,
)
print(f"[judge-only] result_file={result_file} exists={result_file.exists()}")
normalize_online_mind2web_judge_results(result_file=result_file)
PY

echo "[judge-only] score roll-up"
TASKS_META="${TASKS_META:-$REPO/$ASSET_SUBDIR/tasks.json}"
python - "$EVAL_OUTPUT_DIR" "$TASKS_META" <<'PY'
import collections
import json
import sys
from pathlib import Path

eval_dir, tasks_meta = sys.argv[1:]

levels = {}
meta_path = Path(tasks_meta)
if meta_path.is_file():
    for entry in json.loads(meta_path.read_text(encoding="utf-8")):
        levels[entry.get("task_id") or entry.get("id")] = entry.get("level")


def rows(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return list(payload.values()) if isinstance(payload, dict) else payload


total = collections.Counter()
success = collections.Counter()
files = sorted(Path(eval_dir).glob("WebJudge_*auto_eval_results.json"))
if not files:
    raise SystemExit("[judge-only][error] no judge result file found")

for path in files:
    for row in rows(path):
        task_id = row.get("task_id") or row.get("id")
        score = int(row.get("predicted_label") or row.get("score") or 0)
        level = levels.get(task_id, "unknown")
        total[level] += 1
        success[level] += score

for level in ("easy", "medium", "hard", "unknown"):
    if total[level]:
        rate = 100 * success[level] / total[level]
        print(f"[judge-only] {level:8} {success[level]:3}/{total[level]:3} = {rate:5.1f}%")

n = sum(total.values())
s = sum(success.values())
print(f"[judge-only] {'TOTAL':8} {s:3}/{n:3} = {100 * s / n:5.1f}%")
PY

echo "[judge-only] finished rc=0 eval_output=$EVAL_OUTPUT_DIR"
