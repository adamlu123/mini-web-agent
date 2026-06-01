# mini-web-agent — Run Command Guide

## Layout

- Repo: `/home/luyadong/sandbox/mini-web-agent`
- venv: `/home/luyadong/.venv/bin/python`
- Tasks files: `src/miniswewebagent/run/benchmarks/*.json`
  - `om2w_260220.json` — 300 tasks (80 easy / 143 medium / 77 hard)
  - `odysseys.json` — 200 tasks (45 easy / 46 medium / 109 hard)
- Configs: `src/miniswewebagent/config/*.yaml` (referenced by basename, not path)
- Credentials: `/home/luyadong/cred.sh` exports `OPENAI_GATEWAY_API_KEY`, `OPENROUTER_API_KEY`, `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`.
- Entry point (benchmark batch): `python -m miniswewebagent.run.benchmarks.om2w`

## Launch: OpenRouter Qwen3.5-9B, 5 easy tasks in parallel

**Do NOT launch benchmark runs in `screen`.** Launch them as a new pane inside the `inference` window of the current tmux session. Make sure the `inference` window exists first (`tmux new-window -t <session> -n inference` if it doesn't); each run gets its own pane in that window.

```bash
OUT=/home/luyadong/sandbox/mini-web-agent/outputs/default/0520
CFG=best_default_judge_json_openrouter_qwen35_9b.yaml

mkdir -p "$OUT"

# Ensure the inference window exists in the current tmux session, then split a new pane.
tmux has-session 2>/dev/null || { echo "Not in a tmux session"; exit 1; }
tmux list-windows -F '#W' | grep -qx inference || tmux new-window -n inference
tmux split-window -t inference -v \
  "source /home/luyadong/cred.sh && \
   cd /home/luyadong/sandbox/mini-web-agent && \
   /home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
     -c $CFG \
     --task-level easy \
     --limit 5 \
     --workers 5 \
     --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/om2w_260220.json \
     --output-dir $OUT 2>&1 | tee $OUT/run.log; \
   exec bash"
tmux select-layout -t inference tiled
```

Notes:
- The config `best_default_judge_json_openrouter_qwen35_9b.yaml` uses `model_class: openrouter`, `model_name: qwen/qwen3.5-9b`. Confirmed valid OpenRouter slug as of 2026-05-21.
- For a fresh run on the same `OUT` directory, also `rm -rf "$OUT"` before `mkdir -p` to avoid mixing partial trajectories from a prior config.
- `--task-level` is one of `easy|medium|hard`. `--limit N` runs only the first N selected tasks. `--workers` = parallel worker processes.
- Switch to the run pane with `Ctrl-b w` (window list) → pick `inference`, then `Ctrl-b o` to cycle panes.

## Monitor

```bash
tmux list-panes -t inference -F '#{pane_index} #{pane_current_command} #{pane_pid}'
tail -f /home/luyadong/sandbox/mini-web-agent/outputs/default/0520/run.log
ls /home/luyadong/sandbox/mini-web-agent/outputs/default/0520 | wc -l
find /home/luyadong/sandbox/mini-web-agent/outputs/default/0520 \
  -name 'WebJudge_Online_Mind2Web_eval-3.json' | wc -l              # tasks scored
```

Kill: send `C-c` to the run pane, or kill the pane outright:
```bash
tmux kill-pane -t inference.<pane_index>     # specific pane
# or: pkill -f 'miniswewebagent.run.benchmarks.om2w'
```
Then release leftover BB sessions — see the `odyssey-benchmark-run` skill for the REST snippet.

## Other configs

- `best_default_judge_json.yaml` — default phyagi gpt-5.4 (gateway).
- `best_default_judge_json_openrouter_qwen35_9b.yaml` — OpenRouter Qwen3.5-9B, plain agent (no oracle CLI).
- `best_default_judge_json_openrouter_qwen36_cli.yaml` — OpenRouter Qwen3.5-9B with per-task oracle CLI tool injected (`outputs/cli/0426_oracle_cli`). Higher success rate, not a true baseline. (Note: filename says `qwen36_cli` but the configured slug is `qwen/qwen3.5-9b`.)
- `best_default_judge_json_kimi_openrouter.yaml` — OpenRouter Kimi variant.

## Eval / scoring

After a run, per-task judge files land at `<task>/scores/WebJudge_Online_Mind2Web_eval-3.json` with `{"score": 0|1, ...}`. With `judge_enabled: true` in the config, the runner schedules 3 parallel judge runs automatically; results go to `<output_dir>_eval_{1,2,3}/`.

Roll-up by level uses `/home/luyadong/.osagent_eval/om2w/Online_Mind2Web.json` for `task_id → level`.

## Gotchas

- `local_workspace.py` builds subprocess env as `os.environ | self._credential_env | self.config.env | {...}` — **values from `cred.sh` WIN over the live shell env**. If OpenRouter returns 401, fix `cred.sh` itself, not just `export` in your shell.
- Quick OpenRouter smoke test:
  ```bash
  source /home/luyadong/cred.sh
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen/qwen3.5-9b","messages":[{"role":"user","content":"hi"}]}'
  ```
  200 = good. 401 = bad key. 404 = invalid model slug.
- See the `odyssey-benchmark-run` skill for: Browserbase session cleanup after a kill, the per-task session-logging shim, and the phyagi-gateway variant of the same launch flow.
