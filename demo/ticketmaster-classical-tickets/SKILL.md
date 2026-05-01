---
name: ticketmaster-classical-tickets
description: Use a verified Ticketmaster browser automation CLI to find, compare, and recommend classical concert tickets near Seattle.
metadata: {"openclaw":{"always":true,"requires":{"bins":["python"]}}}
---

# Ticketmaster Classical Tickets

Use this skill when the user explicitly mentions `$ticketmaster-classical-tickets` or asks to use the Ticketmaster classical tickets skill by name.

This skill runs a verified Ticketmaster automation CLI. It is specialized for browser-based classical concert ticket comparison tasks, especially the Redmond/Seattle May Day 2026 showcase request. Prefer the CLI over manual browsing whenever the requested task fits these inputs.

## Capability

The CLI can:

- open Ticketmaster through an existing CDP browser
- inspect Seattle classical concert listings
- filter candidates by destination city and date window
- collect visible ticket sections, rows, ticket type, and prices
- compare options against a USD budget
- produce a final recommendation with evidence-oriented screenshots and logs

The CLI does not complete checkout, enter payment information, or fabricate unavailable tickets.

## Default Task

If the user does not override parameters, the CLI uses the verified showcase defaults:

- Start URL: Ticketmaster Seattle classical discovery page
- Origin city: Redmond, Washington
- Destination city: Seattle
- Genre: Classical
- Date window: 2026-05-01 through 2026-05-05
- Budget: 85-140 USD
- Minimum comparable options: 3
- Booking source: Ticketmaster

The verified recommendation at creation time was Seattle Symphony: Xian Conducts Mozart on 2026-05-03 at 2:00 PM at Taper Auditorium, Seattle, WA, with a Ticketmaster resale option around $110.90 before taxes as displayed. Prices and availability can change, so rerun the CLI for current evidence.

## Required Runtime

- A local Chrome/Chromium browser must be available through CDP. The CDP port is required because this CLI attaches to an existing browser session; use `9222` unless the user explicitly chooses another port.
- On macOS, start the browser yourself before running the CLI. Prefer Microsoft Edge if installed, otherwise Google Chrome. Use a persistent `--user-data-dir` so login state survives across runs.
- Use `LOCAL_BROWSER_CDP_URL` if set, otherwise `BROWSER_CDP_URL`, otherwise `http://127.0.0.1:9222`.
- Preserve proxy variables from the caller environment, including `http_proxy`, `https_proxy`, `HTTP_PROXY`, and `HTTPS_PROXY`.
- The script writes run artifacts under this skill directory unless `WORKSPACE_DIR` is already set.

## Procedure

1. Translate the user's request into CLI flags only when they differ from the defaults.

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
   LOCAL_BROWSER_CDP_URL="${LOCAL_BROWSER_CDP_URL:-http://127.0.0.1:9222}" python {baseDir}/ticketmaster_classical_tickets_cli.py
   ```

4. If the user supplied different dates, location, genre, budget, or comparison count, pass explicit CLI flags:

   ```bash
   LOCAL_BROWSER_CDP_URL="${LOCAL_BROWSER_CDP_URL:-http://127.0.0.1:9222}" python {baseDir}/ticketmaster_classical_tickets_cli.py \
     --destination_city "Seattle" \
     --origin_city "Redmond, Washington" \
     --genre "Classical" \
     --start_date "2026-05-01" \
     --end_date "2026-05-05" \
     --budget_min_usd 85 \
     --budget_max_usd 140 \
     --compare_count 3
   ```

5. Read the script output and report:

   - recommended concert
   - date and time
   - venue
   - seat section or ticket tier
   - price
   - booking source
   - compared options
   - checkout notes

6. If the script fails because CDP is unavailable, retry the macOS startup step once and then report the CDP failure. Do not fabricate ticket results.

## Output Contract

The final response should be concise and include the recommended ticket first, followed by the compared alternatives. Mention that Ticketmaster resale prices and availability can change, and that displayed all-in prices may still require tax/final-total review before payment.
