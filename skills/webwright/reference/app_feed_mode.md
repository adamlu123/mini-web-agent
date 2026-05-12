# App/Feed Mode

Default Webwright runs (`/webwright:run`, plain prompt) produce a one-shot
`final_script.py`. CLI tool mode (`/webwright:craft`) produces a reusable
parameterized command. **App/feed mode** (`/webwright:app`) produces a
dual-use script for multisite information-flow tasks: the same
`final_script.py` can run the task once from the CLI and can start a local
Python PWA feed app.

The first version is deliberately local and lightweight: no native app shell,
no Electron/Tauri, no Node build, no system notifications, no background
scheduler, and no user-editable source list. "Push feed" means the page runs
the latest fetch flow when opened and renders the newest unified feed.

Important: app/feed mode changes the **final presentation**, not the underlying
Webwright collection standard. A browser-mediated task must still launch and
use the browser to collect task evidence. The local PWA is the final human UI
that displays the collected results.

The generated script must keep both task lines:

- `python final_script.py --run-once` executes the reusable collection task,
  writes `app/feed.json`, prints the unified JSON, and exits.
- `python final_script.py` starts the local PWA server; the page calls `/run`,
  and `/run` executes the same collection function as `--run-once`.

## When To Use

Use app/feed mode when:

- the user invokes `/webwright:app ...`, or
- the user asks for a local app, PWA, desktop/home-screen icon, or feed UI
  that aggregates latest items from multiple sites, sources, or feeds.

Do not use this mode for a simple one-shot lookup or for a normal reusable CLI
unless the user explicitly wants a local app/feed experience.

## Reference Template

Before writing `final_script.py`, read
[`app_feed_template.py`](app_feed_template.py). The template is the canonical
implementation skeleton for app/feed mode and is part of the contract, not just
an example.

Use it this way:

1. Copy the template structure into the active
   `final_runs/run_<id>/final_script.py`.
2. Replace the configuration block (`APP_TITLE`, `APP_QUERY`, `SOURCES`, item
   limit, and source metadata) with values from the user's task.
3. Replace `fetch_source_items(source)` and, when needed, `sort_items(items)`
   with source-specific exploration findings.
4. Keep the template's import-safe CLI, Python standard library HTTP server,
   `write_app_files()`, cache-first frontend, `/run` endpoint, `/feed.json`
   persistence, `--run-once`, `validate_feed()`, visible demo error/empty
   states, PWA assets, and logging shape unless the task makes a specific
   change necessary.

Do not rewrite the local app/server from scratch when the template fits. The
agent's work should be concentrated on reliable source extraction and
task-specific feed normalization.

## Browser-First Source Collection

App/feed mode must be interpreted as:

```text
normal Webwright source collection -> reusable --run-once -> normalized feed.json -> local PWA UI
```

Do not interpret it as:

```text
skip the browser task -> choose unrelated APIs/RSS feeds -> show a generic UI
```

For tasks whose facts are found through websites, search/listing pages,
shopping/product pages, travel/event/ticket pages, dashboards, forms, or other
UI-mediated workflows, the agent must:

- use Playwright exploration to open the actual source pages;
- capture source evidence screenshots under `final_runs/run_<id>/screenshots/`
  with names such as `exploration_<source>_<evidence>.png` or
  `collection_<source>_<evidence>.png`;
- extract the task-specific fields from those pages or from reliable backing
  requests discovered during that exploration;
- make both `--run-once` and `/run` replay that browser-backed collection flow,
  or an equivalent discovered site-backed request flow that returns the same
  task-specific data;
- render those exact collected items in the final local PWA.

Structured APIs/RSS feeds are acceptable when they are the user's requested
sources, or when source exploration proves they are the site-owned backing data
for the same visible result. They are not acceptable as a substitute for an
unrelated browser task.

Example: for a task like "compare current Apple Mac model prices", the source
evidence screenshots should show Apple or retailer Mac model/pricing pages, and
the feed items should include model, configuration, price, retailer/source,
availability or delivery notes when available, and original links. The final UI
is the comparison surface; it is not the whole task by itself.

## Required Run Layout

Every clean execution of the final script must live in a fresh run folder:

```text
final_runs/run_<id>/
  final_script.py
  final_script_log.txt
  screenshots/
  app/
    index.html
    styles.css
    app.js
    manifest.webmanifest
    feed.json
    icon.svg
```

`final_script.py` must create or refresh the `app/` files before serving the
page. Static files are part of the final artifact and must be inspectable in
the run folder.

## `plan.md` Additions

Before authoring the final script, write both the usual critical points and a
feed-specific contract:

```markdown
# Task
<verbatim task description>

# Feed App Contract
| field | value |
|-------|-------|
| title | <app/feed title> |
| query | <topic/filter/recency summary> |
| sources | <fixed source names and URLs> |
| item shape | title, url, source, published_at, summary, metadata |
| empty state | <visible text/behavior when no items are found> |
| partial failure | <visible text/behavior when one or more sources fail> |
| source evidence | <browser pages/screenshots or source APIs used to collect data> |

# Critical Points
- [ ] CP1: `final_script.py` is import-safe and starts no server/fetch on import.
- [ ] CP2: `python final_script.py --run-once` executes the reusable collection
  task, writes schema-valid `app/feed.json`, and prints the unified JSON.
- [ ] CP3: `python final_script.py` starts a local HTTP server and prints its URL.
- [ ] CP4: app files exist under `app/`, including manifest, icon, and feed JSON.
- [ ] CP5: page load visibly shows a loading state and calls `/run`.
- [ ] CP6: `/run` fetches all fixed sources, writes schema-valid `app/feed.json`,
  and logs source status plus final item count.
- [ ] CP7: task-relevant source screenshots show the original pages/results used
  to collect the feed items.
- [ ] CP8: the read-only UI shows title, update time, source status, feed items,
  original links, and error/empty states when applicable.
```

Add source-specific CPs when the task contains explicit constraints such as a
topic, date range, geographic filter, ranking, or required metadata.

## Final Script Contract

`final_script.py` must be a Python-only local app generator/server.

Required behavior:

1. Importing the module has no side effects. No server starts, no network fetch
   runs, and no files are written at module top level.
2. Running `python final_script.py --help` or an equivalent help path explains
   both task mode (`--run-once`) and app/server mode, including optional
   host/port flags if present.
3. Running `python final_script.py --run-once`:
   - creates or refreshes `app/` files,
   - creates or resets `final_script_log.txt`,
   - executes the reusable collection task without starting a server,
   - updates `app/feed.json`,
   - prints the same schema-valid unified JSON that `/run` returns.
4. Running `python final_script.py`:
   - creates or refreshes `app/` files,
   - creates or resets `final_script_log.txt`,
   - starts a Python standard library HTTP server bound to localhost by
     default,
   - prints a URL like `http://127.0.0.1:<port>/`,
   - serves `index.html`, `styles.css`, `app.js`, `manifest.webmanifest`,
     `feed.json`, and `icon.svg`.
5. `index.html` links the web manifest and icon, loads `app.js`, renders a
   visible loading state immediately, and triggers a `fetch("/run")`.
6. `/run` performs the same multisite collection flow as `--run-once`,
   including Playwright/browser automation when that is how the task data is
   obtained, updates `app/feed.json`, returns the unified feed JSON, and never
   silently drops source failures.
7. The frontend first renders cached `/feed.json` when available, then refreshes
   from `/run` in the background so the first screen is not blank during slow
   source fetches.
8. The frontend renders a read-only feed. It may include links and a refresh
   button only if the refresh simply calls `/run` again. It must not include
   filters, editing, login, notifications, background refresh controls, or
   source-management UI.
9. The server must support `GET` for all static files and `/run`. Supporting
   `HEAD` and `/healthz` is recommended for smoke tests and is included in the
   template.

Use the Python standard library HTTP server stack (`http.server`,
`socketserver`, or equivalents) for serving. Do not add Node, Electron, Tauri,
or extra Python dependencies. Existing Python dependencies in the Webwright
environment, such as Playwright or httpx, may and should be used for actual
source collection when the task requires browser-visible evidence, but prefer
the simplest reliable approach.

## Feed JSON Schema

`app/feed.json` and `/run` must return the same top-level schema:

```json
{
  "title": "string",
  "generated_at": "ISO-8601 string",
  "query": "string",
  "sources": [
    {"name": "string", "url": "string", "status": "ok|error", "item_count": 0}
  ],
  "items": [
    {
      "title": "string",
      "url": "string",
      "source": "string",
      "published_at": "string|null",
      "summary": "string",
      "metadata": {"key": "value"}
    }
  ],
  "errors": [
    {"source": "string", "message": "string"}
  ]
}
```

Schema rules:

- `generated_at` must be an ISO-8601 timestamp generated when `/run` finishes.
- Each configured source appears exactly once in `sources`.
- A source with a caught exception has `status: "error"`, `item_count: 0`, and
  a matching entry in `errors`.
- Each item keeps the original source name and original link.
- `published_at` may be `null` when unavailable, but the field must be present.
- `metadata` must be an object. Use strings for simple metadata values.
- Empty results are represented by `items: []`, visible source statuses, and a
  visible frontend empty state.

## UI Rules

The page must show:

- app/feed title,
- generated/update time,
- source status list with source name, status, item count, and URL,
- feed items,
- each card's title, source, published date when available, summary, metadata,
  and original link,
- visible errors for failed sources,
- visible empty state when no items are returned.

Do not hide partial failures behind console logs. A successful app run can have
some source errors as long as they are shown in the UI and preserved in JSON.
The first viewport must show either cached content, loading status, or a visible
empty/error state; it must not sit blank while `/run` is still fetching.

## Logging

`final_script_log.txt` must be reset at the start of a clean server run and
must include:

- server URL,
- app file generation path,
- `--run-once` start and finish lines,
- `/run` start and finish lines,
- one line per source with source name, URL, status, and item count or error,
- final item count,
- any endpoint or smoke-test evidence the final script records itself.
- browser navigation/extraction evidence for each source when the task uses
  browser-backed collection.

Use the existing Webwright `step <n> action: ...` style for important actions
where possible.

## Screenshots

Screenshots are split into two required evidence classes.

Source evidence screenshots must be saved under
`final_runs/run_<id>/screenshots/` with names such as
`exploration_<source>_<evidence>.png` or
`collection_<source>_<evidence>.png`. They must show the actual source
pages/results used for the user's task. Generic UI screenshots or unrelated
demo pages do not satisfy this requirement.

Final UI screenshots must be saved under
`final_runs/run_<id>/screenshots/final_execution_<step>_<action>.png` and
must include at least:

- loading state before `/run` resolves,
- successful feed state with rendered items when items are found,
- error or empty-state evidence when a source fails or no items are returned.

If all live sources succeed and return items, create deterministic verification
evidence for error/empty rendering by exercising a mock endpoint, a controlled
source-failure branch, or a static frontend state. Do not fake the main feed
results.

## Verification

App/feed mode is complete only when all of the following are true:

1. `plan.md` contains `# Feed App Contract` and `# Critical Points`.
2. `final_script.py` is import-safe.
3. `python final_script.py --help` or equivalent explains both `--run-once`
   and app/server mode.
4. `python final_script.py --run-once` executes the reusable task, prints
   schema-valid JSON, and writes the same schema to `app/feed.json`.
5. Running `python final_script.py` starts a local server and prints the URL.
6. `index.html`, `manifest.webmanifest`, and `feed.json` are reachable through
   the local server.
7. `/run` returns schema-valid JSON and writes the same schema to
   `app/feed.json`.
8. The frontend renders cached `feed.json` before `/run` completes when cached
   items exist.
9. The frontend renders the returned items, source status, update time, and
   error/empty states visibly.
10. Source evidence screenshots are visibly related to the user's task and to
   the feed items rendered in the UI.
11. `final_script_log.txt` records `--run-once`, server URL, per-source status, source
    collection evidence, and final item count.
12. Final screenshots prove loading, success, and error/empty evidence.
13. Every critical point in `plan.md` is checked off using screenshots, logs,
    endpoint responses, or `feed.json`.

If any item fails, fix the script, run it again in a new `final_runs/run_<id+1>/`
folder, and repeat verification.
