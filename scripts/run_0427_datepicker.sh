#!/bin/bash
set -euo pipefail
source /home/luyadong/cred.sh
cd /home/luyadong/sandbox/mini-web-agent
OUT=/home/luyadong/sandbox/mini-web-agent/outputs/cli_fara/0427_datepicker_v3
mkdir -p "$OUT"
exec /home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json_oracle_cli_datepicker.yaml \
    --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/webtail_flight_only.json \
    --no-evaluate \
    --workers 33 \
    --task-level all \
    --output-dir "$OUT" 2>&1 | tee -a "$OUT/run.log"
