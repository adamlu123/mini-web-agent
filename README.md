# mini-swe-webagent

Minimal local-only web agent modeled after mini-swe-agent.

Current implementation scope:

- Local Playwright runtime instead of a shell environment
- Phyagi gateway model client for structured step generation
- Incremental Python Playwright code execution per step
- Browser observations built from:
  - `await page.locator("body").aria_snapshot()`
  - current-step screenshot
  - console output
  - previous message history

## Quick start

1. Install dependencies:

```bash
pip install -e .
playwright install chromium
```

2. Export gateway credentials:

```bash
export OPENAI_GATEWAY_API_KEY=...
export OPENAI_GATEWAY_ENDPOINT=http://gateway.phyagi.net/api/responses
export OPENAI_GATEWAY_TIER=base
```

3. Run a single task:

```bash
mini-web -t "Find the current weather for Vancouver, British Columbia for the next seven days." --start-url https://www.theweathernetwork.com/

set -a && source /Users/lu/Documents/sandbox/cred.sh >/dev/null 2>&1 && source .venv/bin/activate && mini-web -t "Search for a round-trip flight on Singapore Airlines from Singapore to Tokyo departing June 9, 2026 and returning July 4, 2026. If there are no available flights for those dates or the booking is not possible, please indicate that in your answer. IMPORTANT: The task is COMPLETE once the flight search results page shows available flights. Do NOT proceed to seat selection, passenger details, or payment." --start-url "https://www.singaporeair.com/en_UK/us/home" -o "/Users/lu/Documents/sandbox/mini-swe-webagent/outputs/singaporeair-xml-browserbase" -c mini.yaml -c benchmark/webtaibench_xml.yaml -c environment.browserbase_enabled=true -c environment.browserbase_proxies=true -c environment.headless=true -c environment.slow_mo_ms=0 -c environment.browser_timeout_ms=12000 -c environment.browser_navigation_timeout_ms=45000 -c environment.observation_timeout_ms=6000 -c environment.browserbase_timeout_seconds=1800
```

4. Run an Online-Mind2Web task by id:

```bash
mini-web --task-id 871e7771cecb989972f138ecc373107b
```

5. Run the Online-Mind2Web benchmark JSON in batch mode:

```bash
mini-web-om2w --tasks-file /Users/lu/Documents/sandbox/Online-Mind2Web/om2w_260220.json --limit 5
```

By default, `mini-web-om2w` now uses a Browserbase cloud session profile from `benchmark/browserbase.yaml`. It expects `BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID` in the environment.

If you want to force a local browser run instead, disable it explicitly:

```bash
mini-web-om2w --tasks-file /Users/lu/Documents/sandbox/Online-Mind2Web/om2w_260220.json --limit 5 -c mini.yaml -c environment.browserbase_enabled=false
```

The legacy `mini-web-batch` command still works and now points at the same benchmark runner under `src/miniswewebagent/run/benchmarks/`.

For resumable or distributed evaluation, give every invocation the same output directory and batch name:

```bash
mini-web-om2w --tasks-file /path/to/tasks.json --task-level easy+medium \
  --num-shards 2 --shard-index 0 --batch-name om2w-run --resume \
  --output-dir outputs/om2w-run
```

Run the other shard with `--shard-index 1`, then judge the combined output without generating tasks:

```bash
mini-web-om2w --tasks-file /path/to/tasks.json --batch-name om2w-run \
  --judge-only --output-dir outputs/om2w-run
```

Use `--retry-failed` with `--resume` to replace task directories whose `result.json` records `run_exception`.

### Qwen3.5 inference with vLLM

The vLLM integration is inference-only; it does not require LlamaFactory or training code. Install
vLLM in the selected Python environment, then run:

```bash
PY=python TP=4 TASK_LEVEL=easy LIMIT=5 \
  bash scripts/run_vllm_qwen35_om2w.sh
```

The launcher serves `Qwen/Qwen3.5-9B` as `qwen35_9b_base`, waits for
`/v1/models`, and runs OM2W with:

```bash
-c best_default_judge_json_agnostic.yaml -c model_vllm_9b_base.yaml
```

Set `START_VLLM=0 ENDPOINT=http://host:port/v1/chat/completions` to use an existing
OpenAI-compatible server. `MAX_MODEL_LEN`, `MAX_OUTPUT_TOKENS`, and
`MAX_CONTEXT_TOKENS` should describe the same server context budget; the launcher reserves
1,024 tokens for chat-template overhead by default.

## Trace viewer

To inspect runs under `outputs/default` in a browser:

```bash
mini-web-traces --root outputs/default --port 8009
```

Or through the shared utility entry point:

```bash
mini-web-extra trace-viewer --root outputs/default --port 8009
```

The viewer scans each run folder, lists per-task traces, and shows per-step screenshots, actions, thoughts, console output, and ARIA snapshots.

To relaunch it with a shareable Cloudflare link from the repo root, use:

```bash
PYTHONPATH=src python -m miniswewebagent.run.utilities.trace_viewer --root outputs/default --host 127.0.0.1 --port 52869
```

Then, in a second terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:52869
```

Share the `https://*.trycloudflare.com` URL printed by `cloudflared`.

To stop the shared viewer and free the local port, kill all matching viewer and tunnel PIDs with:

```bash
pids=$(ps -eo pid=,command= | awk '/miniswewebagent\.run\.utilities\.trace_viewer --root outputs\/default --host 127\.0\.0\.1 --port 52869/ || /cloudflared tunnel --url http:\/\/127\.0\.0\.1:52869/ {print $1}'); [ -n "$pids" ] && kill $pids
```

## Review viewer

To inspect sandbox runs alongside judge outputs:

```bash
mini-web-review --runs-root outputs/sandbox --judge-root om2w_judge --port 8010
```

Or use the repo helper that mirrors the remote viewer workflow:

```bash
bash scripts/start_review_viewer.sh 8010
```

To launch it on a remote machine and keep it localhost-only:

```bash
bash scripts/start_remote_review_viewer.sh 52870
```

Then, in a second terminal on that same machine:

```bash
bash scripts/start_public_tunnel.sh 52870
```

Share the `https://*.trycloudflare.com` URL printed by `cloudflared`.

## Docs

- Main execution flow: [docs/execution-flow.md](docs/execution-flow.md)
