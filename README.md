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
export OPENAI_GATEWAY_ENDPOINT=http://gateway.phyagi.net/api/chat/completions
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
