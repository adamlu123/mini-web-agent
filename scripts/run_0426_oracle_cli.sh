#!/bin/bash
set -euo pipefail
source /home/luyadong/cred.sh
cd /home/luyadong/sandbox/mini-web-agent
OUT=/home/luyadong/sandbox/mini-web-agent/outputs/cli/0426_oracle_cli
mkdir -p "$OUT"
exec /home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json_oracle_cli.yaml \
    --workers 300 \
    --task-level all \
    --output-dir "$OUT" 2>&1 | tee -a "$OUT/run.log"
