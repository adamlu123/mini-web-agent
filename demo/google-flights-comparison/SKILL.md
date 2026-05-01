---
name: google-flights-comparison
description: Use a verified Google Flights browser automation CLI to compare round-trip flight options and recommend the best itinerary.
metadata: {"openclaw":{"always":true,"requires":{"bins":["python"]}}}
---

# Google Flights Comparison

Use this skill when the user explicitly mentions `$google-flights-comparison` or asks to use the Google Flights comparison skill by name.

This skill runs a verified Google Flights automation CLI for browser-based round-trip economy flight comparison tasks. Prefer the CLI over manual browsing whenever the user gives concrete origin, destination, dates, budget, and comparison requirements.

## Capability

The CLI can:

- open Google Flights through an existing CDP browser
- set origin, destination, country, dates, cabin, passengers, budget, and comparison count
- collect multiple flight options
- rank options under budget with preference for shorter layovers and reasonable total travel time
- produce a final recommendation with evidence-oriented output

The CLI does not book tickets, enter payment information, or fabricate unavailable fares.

## Default Task

If the user does not override parameters, the CLI uses the verified showcase defaults:

- Origin: Hong Kong
- Destination: Rio de Janeiro, Brazil
- Outbound date: 2026-05-03
- Return date: 2026-05-11
- Cabin: Economy
- Adults: 1
- Budget: under 20,000 HKD
- Minimum comparable options: 3
- Preference: shorter layovers and reasonable total travel time

## Required Runtime

- A local Chrome/Chromium browser must be available through CDP. The CDP port is required because this CLI attaches to an existing browser session; use `9222` unless the user explicitly chooses another port.
- On macOS, start the browser yourself before running the CLI. Prefer Microsoft Edge if installed, otherwise Google Chrome. Use a persistent `--user-data-dir` so login state survives across runs.
- Use `LOCAL_BROWSER_CDP_URL` if set, otherwise `BROWSER_CDP_URL`, otherwise `http://127.0.0.1:9222`.
- Preserve proxy variables from the caller environment, including `http_proxy`, `https_proxy`, `HTTP_PROXY`, and `HTTPS_PROXY`.
- The script writes run artifacts under this skill directory unless `WORKSPACE_DIR` is already set.

## Procedure

1. Translate the user's request into CLI flags. If the user provides any city, country, date, budget, cabin, passenger count, or minimum option count, those values MUST be passed as explicit flags rather than relying on defaults.

2. Ensure a local macOS CDP browser is running. This starts a logged-in, reusable browser profile if nothing is listening on the port:

   ```bash
   export LOCAL_BROWSER_CDP_URL="${LOCAL_BROWSER_CDP_URL:-${BROWSER_CDP_URL:-http://127.0.0.1:9222}}"
   CDP_PORT="${LOCAL_BROWSER_CDP_PORT:-$(printf '%s\n' "$LOCAL_BROWSER_CDP_URL" | sed -E 's#.*:([0-9]+).*#\1#')}"
   if ! lsof -nP -iTCP:"${CDP_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
     BROWSER_PROFILE="${LOCAL_BROWSER_USER_DATA_DIR:-$HOME/.cache/mini-web-agent/edge-profile}"
     mkdir -p "$BROWSER_PROFILE"
     if [ -d "/Applications/Microsoft Edge.app" ]; then
       open -na "Microsoft Edge" --args \
         --remote-debugging-address=127.0.0.1 \
         --remote-debugging-port="${CDP_PORT}" \
         --user-data-dir="$BROWSER_PROFILE"
     elif [ -d "/Applications/Google Chrome.app" ]; then
       open -na "Google Chrome" --args \
         --remote-debugging-address=127.0.0.1 \
         --remote-debugging-port="${CDP_PORT}" \
         --user-data-dir="$BROWSER_PROFILE"
     else
       echo "Install Microsoft Edge or Google Chrome for local CDP browser automation." >&2
       exit 1
     fi
     for _ in 1 2 3 4 5; do
       curl -fsS "$LOCAL_BROWSER_CDP_URL/json/version" >/dev/null 2>&1 && break
       sleep 1
     done
   fi
   ```

3. Run the skill CLI from this skill directory:

   ```bash
   LOCAL_BROWSER_CDP_URL="${LOCAL_BROWSER_CDP_URL:-http://127.0.0.1:9222}" python {baseDir}/google_flights_comparison_cli.py
   ```

4. If the user supplied different cities, dates, budget, cabin, or passenger count, pass explicit CLI flags:

   ```bash
   LOCAL_BROWSER_CDP_URL="${LOCAL_BROWSER_CDP_URL:-http://127.0.0.1:9222}" python {baseDir}/google_flights_comparison_cli.py \
     --origin "Hong Kong" \
     --destination "Jeju Island" \
     --country "South Korea" \
     --outbound_date "2026-08-08" \
     --return_date "2026-08-14" \
     --max_price_hkd 20000 \
     --cabin "Economy" \
     --adults 1 \
     --min_options 3 \
     --prefer_short_layovers true
   ```

5. Read the script output and report:

   - compared options
   - final recommendation
   - price
   - dates
   - route
   - layovers
   - booking source

6. If the script fails because CDP is unavailable, retry the macOS startup step once and then report the CDP failure. Do not fabricate flight results.

## Output Contract

The final response should be concise and include the recommended itinerary first, followed by the compared alternatives. Mention that prices and availability can change on Google Flights.
