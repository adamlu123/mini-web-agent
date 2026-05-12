---
description: Build a local PWA information feed app for a multisite Webwright task.
argument-hint: <multisite information feed task with concrete sources/topic/filters>
---

You are operating as the Webwright agent in **app/feed mode**. First read
the `SKILL.md` of the `webwright` skill (the parent directory of this
`commands/` folder) and the `reference/app_feed_mode.md` next to it,
then read `reference/app_feed_template.py`. Build the following multisite
information feed task as a local Python PWA app by copying the template
structure and replacing only the task-specific configuration, fetchers, sorting,
and item metadata:

$ARGUMENTS

Steps:

1. **Identify the feed contract.** Extract the fixed sources, topic,
   recency/sort rules, and any filters from the task. In app/feed mode these
   are hard-coded into `final_script.py`; do not ask the user for runtime
   source editing or login.

2. **Write `plan.md`.** Include a `# Feed App Contract` section with the app
   title, query, fixed sources, item fields, empty-state behavior, and
   expected partial-failure behavior. Also include the usual
   `# Critical Points` checklist.

3. **Explore sources.** Use Webwright's normal exploration workflow to solve
   the user's concrete task before building the UI. For browser-mediated tasks
   such as product comparison, listings, shopping, travel, ticket/event lookup,
   or search-result review, launch Playwright, open the actual source pages,
   and save task-relevant source evidence screenshots. Use structured feeds or
   APIs only when the task asks for them or exploration proves they are the
   site's reliable backing data for the same visible results. Keep screenshots
   and notes in the workspace.

4. **Author `final_script.py`** in a fresh `final_runs/run_<id>/`:
   - It is a dual-use Python script: `--run-once` runs the reusable Webwright
     collection task and prints JSON; default execution starts the local PWA
     app/server.
   - It follows `reference/app_feed_template.py`: preserve its import-safe
     CLI, `--run-once`, standard library HTTP server, PWA asset generation,
     cache-first frontend, `/run`, `/feed.json`, schema validation, demo
     error/empty states, and logging shape.
   - It uses the Python standard library HTTP server for local serving.
   - It creates `app/index.html`, `app/styles.css`, `app/app.js`,
     `app/manifest.webmanifest`, `app/feed.json`, and `app/icon.svg`.
   - Running `python final_script.py --run-once` executes the task without
     starting the server, writes `app/feed.json`, and prints the unified JSON.
   - Running `python final_script.py` starts the server and prints a URL like
     `http://127.0.0.1:<port>/`.
   - `index.html` shows cached `feed.json` immediately when available, calls
     `/run` in the background, then renders the updated read-only feed UI.
   - `/run` performs the same multisite collection workflow as `--run-once`,
     including browser automation when that is how the task data is obtained,
     updates `app/feed.json`, and returns JSON matching the schema in
     `reference/app_feed_mode.md`.
   - `final_script.py` is side-effect-free at import time: importing it starts
     no server, performs no fetch, and writes no files.

5. **Verify the app.**
   - Run `python final_runs/run_<id>/final_script.py --help` or an equivalent
     help path and confirm the user can see how to start the app.
   - Run an import-safety smoke test.
   - Run `python final_runs/run_<id>/final_script.py --run-once`, validate the
     printed schema, and confirm `app/feed.json` matches.
   - Start the server, open the printed local URL, and confirm
     `index.html`, `manifest.webmanifest`, and `feed.json` are reachable.
   - Trigger `/run` and verify the returned schema, source statuses, item
     count, and visible feed items.
   - Capture source evidence screenshots from the actual task pages/results,
     plus final UI screenshots for loading state, successful feed state, and
     error or empty-state evidence when applicable.

6. **Self-verify** every critical point against `final_script_log.txt`,
   `app/feed.json`, endpoint responses, and saved screenshots. If any CP fails,
   diagnose, fix the script preserving app/feed shape, re-run in
   `final_runs/run_<id+1>/`, and re-verify.

Refer to `reference/app_feed_mode.md` for the complete app contract and
`reference/playwright_patterns.md` for the Playwright screenshot workflow.
