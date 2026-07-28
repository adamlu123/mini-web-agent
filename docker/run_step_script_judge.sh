#!/usr/bin/env bash

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT not set}"
: "${JOB_NAME:?JOB_NAME not set}"
: "${TRAJECTORIES_DIR:?TRAJECTORIES_DIR not set}"
: "${JUDGE_OUTPUT_DIR:?JUDGE_OUTPUT_DIR not set}"

REPO="${REPO:-$DATA_ROOT/runs/$JOB_NAME/mini-web-agent}"
CREDS_FILE="${CREDS_FILE:-/run/secrets/webchain-sampling/cred.sh}"
JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
JUDGE_SCORE_THRESHOLD="${JUDGE_SCORE_THRESHOLD:-3}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-150}"
TASKS_FILE="${TASKS_FILE:-src/miniswewebagent/run/benchmarks/om2w_260220.json}"
SUMMARY_PATH="${SUMMARY_PATH:-$JUDGE_OUTPUT_DIR/run_summary_judge.json}"
REPO_LINK=/home/luyadong/sandbox/mini-web-agent

test -d "$REPO"
test -d "$TRAJECTORIES_DIR"
test -f "$CREDS_FILE"
test -f "$REPO/$TASKS_FILE"
mkdir -p "$(dirname "$REPO_LINK")" "$JUDGE_OUTPUT_DIR"
if [[ -L "$REPO_LINK" ]]; then
  unlink "$REPO_LINK"
elif [[ -e "$REPO_LINK" ]]; then
  echo "[error] $REPO_LINK exists and is not a symlink." >&2
  exit 1
fi
ln -s "$REPO" "$REPO_LINK"
cd "$REPO"

TASK_COUNT=$(find "$TRAJECTORIES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
STEP_TASKS=$(find "$TRAJECTORIES_DIR" -mindepth 2 -maxdepth 2 -type d -name steps | wc -l)
SAMPLE_TASK=""
LATEST_RUN=""
while IFS= read -r task_dir; do
  [[ -d "$task_dir/steps" && -d "$task_dir/final_runs" ]] || continue
  run_name=$(find "$task_dir/final_runs" -mindepth 1 -maxdepth 1 -type d -name "run_*" -printf "%f\n" |
    sort -V | tail -1)
  [[ -n "$run_name" && -d "$task_dir/final_runs/$run_name/screenshots" ]] || continue
  SAMPLE_TASK="$task_dir"
  LATEST_RUN="$run_name"
  break
done < <(find "$TRAJECTORIES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

[[ "$TASK_COUNT" -gt 0 && "$STEP_TASKS" -gt 0 && -n "$SAMPLE_TASK" ]]
STEP_COUNT=$(find "$SAMPLE_TASK/steps" -maxdepth 1 -type f -name "*.sh" | wc -l)
SHOT_COUNT=$(find "$SAMPLE_TASK/final_runs/$LATEST_RUN/screenshots" -maxdepth 1 -type f -iname "*.png" | wc -l)
echo "[mode-d] tasks=$TASK_COUNT tasks_with_steps=$STEP_TASKS sample=$(basename "$SAMPLE_TASK") latest_run=$LATEST_RUN sample_steps=$STEP_COUNT sample_screenshots=$SHOT_COUNT"
[[ "$STEP_COUNT" -gt 0 && "$SHOT_COUNT" -gt 0 ]]

python -c "import backoff, httpx, openai; from PIL import Image" ||
  python -m pip install --no-cache-dir openai httpx backoff pillow

set +u
source "$CREDS_FILE"
set -u
: "${OPENAI_GATEWAY_API_KEY:?OPENAI_GATEWAY_API_KEY not set by $CREDS_FILE}"
: "${OPENAI_GATEWAY_ENDPOINT:?OPENAI_GATEWAY_ENDPOINT not set by $CREDS_FILE}"
unset OPENAI_API_BACKUP_KEY
export OPENAI_API_KEY="$OPENAI_GATEWAY_API_KEY"

exec python scripts/eval_with_original_om2w.py \
  --trajectories_dir "$TRAJECTORIES_DIR" \
  --output_path "$JUDGE_OUTPUT_DIR" \
  --summary_path "$SUMMARY_PATH" \
  --tasks_file "$TASKS_FILE" \
  --action_history_source step_scripts \
  --model "$JUDGE_MODEL" \
  --score_threshold "$JUDGE_SCORE_THRESHOLD" \
  --num_worker "$JUDGE_NUM_WORKERS"
