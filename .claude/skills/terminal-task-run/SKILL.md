---
name: terminal-task-run
description: Generate agent trajectories on RST terminal tasks (Recursive Synthetic Terminal Tasks) with the Docker harness and the phyagi gateway — prepare the task pool, launch a band, monitor, score with each task's private tests, clean up. Use when asked to run, launch, monitor or score terminal (non-web) task generation, or when working with terminal_docker, terminal_rst.yaml, or run.benchmarks.rst.
---

# RST terminal task runs

Each task ships its own `environment/Dockerfile` and its own private `tests/`. The
agent runs inside the task's container; `tests/test.sh` runs afterwards and produces
the label (`score` 1/0). Self-reflection does not affect the label.

Target host: Linux x86_64, Docker daemon, `/home/luyadong/.venv/bin/python`,
`source /home/luyadong/cred.sh` for `OPENAI_GATEWAY_API_KEY`.

## 1. Prepare the task pool (once per machine)

```bash
source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python \
  .claude/skills/terminal-task-run/scripts/prepare_rst_tasks.py \
  --dest /home/luyadong/data/rst
```

Downloads `Zhongzhi1228/Recursive-Task-Synthesis` (~3.9 GB, public), dedups rewrite
variants (37,484 → 12,010), keeps the shortest 5,000 by reference-solution length,
splits them into 10 bands of 500 (`g01` easiest … `g10` hardest), extracts the packages
and drops the ~583 multi-container tasks. Deterministic. Needs `huggingface_hub`,
`pyarrow`, `pandas`.

```
/home/luyadong/data/rst/selection/
  tasks/tasks/<task_id>/   ← tasks_root
  g01.jsonl … g10.jsonl    ← ~440 tasks each
  all.jsonl
```

## 2. Launch one band

Open a pane in the `inference` window of the current tmux session (no `screen`).

```bash
BAND=g01
DATA=/home/luyadong/data/rst/selection
OUT=/home/luyadong/sandbox/mini-web-agent/outputs/terminal/rst_${BAND}
WORKERS=${WORKERS:-8}              # agent concurrency; override per run, e.g. WORKERS=32
BUILD_WORKERS=${BUILD_WORKERS:-4}  # concurrent docker builds; keep well below WORKERS

mkdir -p "$OUT"
tmux list-windows -F '#W' | grep -qx inference || tmux new-window -n inference
tmux split-window -t inference -v \
  "source /home/luyadong/cred.sh && \
   cd /home/luyadong/sandbox/mini-web-agent && \
   /home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.rst \
     -c generation/terminal_rst.yaml \
     -c generation/judge_phyagi.yaml \
     -c environment.tasks_root=$DATA/tasks/tasks \
     -c environment.build_timeout_seconds=1200 \
     -c agent.require_self_reflection_success=false \
     -c agent.step_limit=50 \
     --tasks-file $DATA/${BAND}.jsonl \
     --workers $WORKERS --build-workers $BUILD_WORKERS --prune-every 25 \
     --output-dir $OUT 2>&1 | tee $OUT/run.log; \
   exec bash"
tmux select-layout -t inference tiled
```

Smoke-test first with `--limit 2 --workers 2` and confirm two `summary.json` files
with a non-null `score`.

- `generation/judge_phyagi.yaml` is required: without it self-reflection falls back
  to TRAPI Kimi and fails on every call.
- `environment.tasks_root` is required: the yaml default is a dev path.
- `require_self_reflection_success=false` turns off the completion gate. On 20 g01
  tasks the gate produced the same positives as no gate at 2–3× the cost; labels come
  from `tests/test.sh` either way. Leave it off for data generation.
- `--workers` is agent concurrency (gateway-bound); `--build-workers` caps concurrent
  image builds (host-bound). Each container asks for ~2 GB; size workers by RAM.
  If the user names a worker count, set `WORKERS` (and `BUILD_WORKERS`) above instead
  of editing the command; the defaults are 8 / 4. Rate-limit retries are per model
  call (`MAX_RATE_LIMIT_RETRIES` in `models/phyagi_model.py`, 10 with linear backoff
  capped at 30 s); a task whose retries are exhausted lands in `summary.json` as
  `error`, not a crash, so large `WORKERS` mostly costs samples, not the batch.
- For local Azure runs replace `judge_phyagi.yaml` with `model_azure_gpt54.yaml` +
  `judge_azure_gpt54.yaml`.

## 3. Monitor

```bash
tail -f $OUT/run.log
find $OUT -name summary.json | wc -l
docker ps --filter name=rst- --format '{{.Names}} {{.Status}}'
```

Per task in `$OUT/<task_id>/`: `trajectory.json` (the SFT data), `summary.json`
(steps, exit_status, score, seconds), `verifier_log.txt`, `steps/`, `logs/`.
`$OUT/batch_summary.json` is the roll-up.

## 4. Score

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("OUT_DIR")
rows = [json.loads(p.read_text()) for p in root.glob("*/summary.json")]
scored = [r for r in rows if isinstance(r.get("score"), int)]
print(f"{sum(r['score'] for r in scored)}/{len(scored)} passed",
      " skipped:", sum(1 for r in rows if r.get("skipped")),
      " errors:", sum(1 for r in rows if r.get("error")))
PY
```

Keep trajectories with `score == 1`. `score` is the private tests' verdict; `null`
means the tests produced no reward (infrastructure), `skipped` means a multi-container
task. `exit_status` is not a label.

If a band is slow, `scripts/diagnose_throughput.py --output-dir $OUT --workers N`
separates build time, step-limit churn, gateway latency and disk pressure.

## 5. Kill and clean up

```bash
tmux send-keys -t inference.<pane_index> C-c      # signals the whole process group
docker ps -a --filter name=rst- -q | xargs -r docker rm -f
docker builder prune -f --keep-storage 20GB
```

Do not use `pkill -f 'benchmarks[.]rst'` alone: it reaches only the parent (workers
now exit on their own within seconds of the parent dying, but `C-c` is the right
tool). Only remove `rst-` containers; the daemon is shared.
