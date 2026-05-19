#!/bin/bash
set -euo pipefail
source /home/luyadong/cred.sh
cd /home/luyadong/sandbox/mini-web-agent
exec /home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0425_all_addtask
