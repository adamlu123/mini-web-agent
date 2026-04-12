# Mini Web Agent Execution Flow

This document explains the complete `mini-web` runtime path and lists the main code, files, and outputs produced at each stage.

## 1. High-level pipeline

```text
+----------------------------------------------------------------------------------+
| Input                                                                            |
|   CLI: mini-web                                                                  |
|   - task / task_id / tasks_file                                                  |
|   - start_url                                                                    |
|   - config specs                                                                 |
|   - output_dir / debug                                                           |
|   Code: run/mini.py, config/mini.yaml, tasks/om2w.py                             |
|   Files: YAML config files, benchmark JSON when task_id is used                  |
+----------------------------------------------------------------------------------+
                                       |
                                       v
+----------------------------------------------------------------------------------+
| Stage A: Config resolution and component wiring                                  |
|   Merge configs, resolve task text and start_url, create output paths            |
|   Build model + environment + agent                                              |
|   Code: run/mini.py, models/__init__.py, environments/__init__.py,               |
|         agents/__init__.py                                                       |
|   Files: output_dir/, trajectory.json target, runtime_errors.jsonl target        |
|   Output: live runtime objects and resolved output directory                     |
+----------------------------------------------------------------------------------+
                                       |
                                       v
+----------------------------------------------------------------------------------+
| Stage B: Browser environment preparation                                         |
|   Start Playwright runtime                                                       |
|   Connect local Chromium or Browserbase                                          |
|   Open start_url and wait for rendered page readiness                            |
|   Code: environments/local_browser.py                                            |
|   Files: output_dir/, runtime_errors.jsonl                                       |
|   Output: prepared page/context/browser state                                    |
+----------------------------------------------------------------------------------+
                                       |
                                       v
+----------------------------------------------------------------------------------+
| Stage C: Agent bootstrap                                                         |
|   Render system and user templates                                               |
|   Seed the conversation history                                                  |
|   Code: agents/default.py, config/mini.yaml                                      |
|   Files: trajectory.json                                                         |
|   Output: initial message list for the model                                     |
+----------------------------------------------------------------------------------+
                                       |
                                       v
+----------------------------------------------------------------------------------+
| Stage D: Main loop                                                               |
|   Query model -> parse step -> execute action -> capture observation             |
|   Repeat until done=true or step limit / exception                               |
|   Code: agents/default.py, models/phyagi_model.py,                               |
|         environments/local_browser.py                                            |
|   Files: steps/step_XXXX.py, script.py, screenshots/step_XXXX.png,               |
|          debug/steps/step_XXXX.json, debug/steps.md, trajectory.json             |
|   Outputs: per-step code, observations, updated conversation state               |
+----------------------------------------------------------------------------------+
                                       |
                                       v
+----------------------------------------------------------------------------------+
| Stage E: Finalization and artifact export                                        |
|   Close browser runtime                                                          |
|   Export result bundle for inspection and evaluation                             |
|   Code: run/mini.py, utils/om2w_eval.py                                          |
|   Files: result.json, trajectory/, trajectory.json, runtime_errors.jsonl         |
|   Output: stable run artifact bundle                                             |
+----------------------------------------------------------------------------------+
                                       |
                                       v
+----------------------------------------------------------------------------------+
| Stage F: Optional inspection                                                     |
|   Trace viewer reads the output directory and renders step-by-step traces        |
|   Code: run/utilities/trace_viewer.py, viewer/                                   |
|   Files: outputs/<run>/..., trajectory.json, result.json, screenshots/           |
|   Output: local browser UI for debugging and review                              |
+----------------------------------------------------------------------------------+
```

## 2. Stage-by-stage flow

### Stage A: Config resolution and component wiring

The entrypoint is `mini-web`, which maps to `miniswewebagent.run.mini:app`.

`run_one(...)` does the following:

- loads one or more config specs
- merges them into one runtime config
- resolves either:
  - an ad hoc task from `--task`, or
  - a benchmark task from `--task-id` plus `--tasks-file`
- computes a timestamped output directory
- injects derived paths into config for the agent, model, and environment
- constructs the three core runtime objects

Code:

- `src/miniswewebagent/run/mini.py`
- `src/miniswewebagent/models/__init__.py`
- `src/miniswewebagent/environments/__init__.py`
- `src/miniswewebagent/agents/__init__.py`
- `src/miniswewebagent/tasks/om2w.py`

Files read:

- one or more YAML config specs, typically `src/miniswewebagent/config/mini.yaml`
- benchmark task JSON when `--task-id` is used

Files defined at this stage:

- `output_dir/`
- `output_dir/trajectory.json`
- `output_dir/runtime_errors.jsonl`

Output:

- resolved task text
- resolved start URL
- resolved output directory
- constructed `PhyagiModel`, `LocalBrowserEnvironment`, and `DefaultAgent`

### Stage B: Browser environment preparation

Before the agent starts its reasoning loop, the runner calls `env.prepare(...)`.

`LocalBrowserEnvironment.prepare(...)` then:

- stores the prepared task metadata
- starts Playwright if needed
- launches local Chromium or connects to Browserbase over CDP
- reuses or creates a browser context and page
- navigates to `start_url` when present
- waits for the page to become stable enough for observation

Code:

- `src/miniswewebagent/environments/local_browser.py`

Files touched:

- `output_dir/`
- `output_dir/runtime_errors.jsonl`

Outputs:

- live `playwright`, `browser`, `context`, and `page` handles
- browser session metadata when Browserbase is enabled
- initial ready-to-observe page state

### Stage C: Agent bootstrap

`DefaultAgent.run(...)` resets internal state and seeds the conversation with two messages:

- a system message from `agent.system_template`
- a user message from `agent.instance_template`

The default prompt contract requires the model to return one XML step at a time with:

- `thought`
- `python_code`
- `done`
- `final_response`

Code:

- `src/miniswewebagent/agents/default.py`
- `src/miniswewebagent/config/mini.yaml`

Files touched:

- `output_dir/trajectory.json`

Output:

- initial message history containing task, start URL, and runtime prompt instructions

### Stage D: Main loop

This is the core agent cycle.

```text
message history
    |
    v
model.query(messages)
    |
    +--> gateway request serialized from system/user/assistant messages
    |
    v
parse response
    |
    +--> thought
    +--> python_code
    +--> done
    +--> final_response
    |
    v
if done=true
    |
    +--> emit exit message
    |
    v
else execute action in browser
    |
    +--> persist step code
    +--> run async Playwright snippet
    +--> wait for page settle
    +--> capture screenshot + ARIA snapshot + console + URL + title
    |
    v
format observation as next user message
    |
    +--> include text summary
    +--> include screenshot image when available
    |
    v
repeat
```

The work is split across three modules.

#### D1. Model query and parse

`PhyagiModel.query(...)`:

- serializes the message history into the gateway request schema
- posts to the configured Responses-compatible gateway endpoint
- retries transient and rate-limit failures
- parses XML or JSON output
- converts `python_code` into an action list for the agent

Code:

- `src/miniswewebagent/models/phyagi_model.py`

Files touched:

- `output_dir/runtime_errors.jsonl`

Outputs:

- assistant message with:
  - `content`: the step thought
  - `extra.actions`: generated Playwright code
  - `extra.done`
  - `extra.final_response`

#### D2. Browser step execution

`LocalBrowserEnvironment.execute(...)` runs the generated step.

For each step it:

- ensures the runtime still exists
- waits for any Browserbase captcha pause to clear
- writes the code to `steps/step_XXXX.py`

- appends the same code to `script.py`
- executes the step as async Python with live browser objects
- waits for observation readiness
- captures a screenshot and ARIA snapshot
- records console output and exceptions

Code:

- `src/miniswewebagent/environments/local_browser.py`

Files written per step:

- `output_dir/steps/step_XXXX.py`
- `output_dir/script.py`
- `output_dir/screenshots/step_XXXX.png`

Outputs:

- observation object with:
  - `success`
  - `exception`
  - `url`
  - `title`
  - `screenshot_path`
  - `aria_snapshot`
  - `console_output`

#### D3. Observation feedback and debug persistence

After execution, the agent and model convert the observation into the next turn's input.

`DefaultAgent` also persists debug-friendly step artifacts.

Code:

- `src/miniswewebagent/agents/default.py`
- `src/miniswewebagent/models/phyagi_model.py`

Files written during the loop:

- `output_dir/debug/steps/step_XXXX.json`
- `output_dir/debug/steps.md`
- `output_dir/trajectory.json`

Outputs:

- next user observation message
- updated serialized trajectory
- step-by-step debug summary for later inspection

### Stage E: Finalization and artifact export

After the loop exits, `run_one(...)` closes the environment and always attempts to export a consistent result bundle.

This happens even when the run failed partway through.

The export layer:

- copies or normalizes per-step screenshots into `trajectory/`
- extracts action history from debug steps or falls back to `trajectory.json`
- writes `result.json` with task metadata, submission, thoughts, actions, and screenshot paths

Code:

- `src/miniswewebagent/run/mini.py`
- `src/miniswewebagent/utils/om2w_eval.py`

Files written:

- `output_dir/result.json`
- `output_dir/trajectory/0_full_screenshot.png`
- `output_dir/trajectory/1_full_screenshot.png`
- `output_dir/trajectory/...`

Files consumed:

- `output_dir/screenshots/step_XXXX.png`
- `output_dir/debug/steps/step_XXXX.json`
- `output_dir/trajectory.json`

Outputs:

- stable run summary in `result.json`
- normalized screenshot bundle for evaluation or trace viewing

### Stage F: Optional inspection via trace viewer

The trace viewer is not part of the generation loop, but it is the main consumer of the produced artifacts.

It scans the outputs root, loads each run folder, and renders per-step screenshots, actions, thoughts, and raw result JSON.

Code:

- `src/miniswewebagent/run/utilities/trace_viewer.py`
- `src/miniswewebagent/viewer/index.html`
- `src/miniswewebagent/viewer/app.js`
- `src/miniswewebagent/viewer/styles.css`

Files consumed:

- `outputs/<run>/trajectory.json`
- `outputs/<run>/result.json`
- `outputs/<run>/screenshots/`
- `outputs/<run>/debug/steps/`

Output:

- local trace-inspection UI served by `mini-web-traces`

## 3. What gets written to disk

A typical run directory looks like this:

```text
<output_dir>/
  trajectory.json
  result.json
  runtime_errors.jsonl
  script.py
  steps/
    step_0001.py
    step_0002.py
    ...
  screenshots/
    step_0001.png
    step_0002.png
    ...
  debug/
    steps.md
    steps/
      step_0001.json
      step_0002.json
      ...
  trajectory/
    0_full_screenshot.png
    1_full_screenshot.png
    ...
```

## 4. Mental model

The shortest correct mental model for the repository is:

```text
runner
  -> resolves task and config
  -> builds model + browser environment + agent
  -> asks the model for one browser step
  -> executes that step in a live Playwright session
  -> captures observation and feeds it back to the model
  -> repeats until done
  -> closes the runtime and exports a stable artifact bundle
```

Everything else in the repository supports that loop.