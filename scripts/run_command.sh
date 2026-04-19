source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_iterative_judge.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/iterative/0418_all_best_repro