# OSWorld model-run comparison

The mild score difference between GPT-5.6 Sol and GPT-6 Astra is smaller than the harness mismatch. Use fixed-105 scores for a consistent headline; Astra's 69.10% is evaluated-only.

## Scores and run conditions

| Model | Pin | Harness | Provider / width | Evaluated | Score sum | Fixed-105 | Evaluated-only |
|---|---|---|---|---:|---:|---:|---:|
| GPT-5.5 xhigh | `8b6a87da4632b0976bc6ce4fed94c08e425dffa3` | Direct linear Actor | AIServices / 48 | 103/105 | 63.570957 | **60.543769%** | 61.719376% |
| GPT-5.6 Sol max | `592db6dde369518741f46afc69a298bdddce94ee` | E Actor + defective Source Binder | TRAPI 70/16 + AIServices 35/8 | 103/105 | 70.562320 | **67.202209%** | 68.507107% |
| GPT-6 Astra max | `678d20f39ee374543a57f2294199dd152726c98b` | Direct linear Actor | AIServices / 12 | 102/105 | 70.480140 | **67.123943%** | **69.098177%** |

All used OSWorld V2's same 105-task subset, benchmark source `43997ed9a88b81151bbb496f1fd1f3b9febf312c`, the repaired QCOW2, terminal-only interaction, a 300-call task ceiling, compaction every 40 calls, and no quality retries. GPT-5.5 and Astra are the closest harness comparison. Sol used a shorter Actor prompt, mixed providers, and a Binder contract that replaced useful reports with forced-invalid fallbacks on 90/105 attempts. On the 101 tasks scored by both Sol and Astra, Astra was +0.9904 percentage points.

**Direct-Sol availability:** no completed full-105 GPT-5.6 Sol Max direct-Actor run was found. The only completed direct condition is the 14-task setup-semantics panel at commit `e1ddedac415f143de91de3dcead26a8a0ccbd500`: 14/14 evaluated, score sum `8.357842040918678`, mean **59.698872%**, TRAPI width 14. The older **67.327979%** composite at `2ef21b1a` was also orchestrated—it used Binder, Critic, and Verifier—not direct Actor.

## GPT-5.5 baseline lineage

The September 5 and September 10 GPT-5.5 runs are the same model-visible condition, but not the same runner implementation.

| Condition | Older canonical pin | Recent merged-master baseline |
|---|---:|---:|
| Commit | `8b6a87da4632b0976bc6ce4fed94c08e425dffa3` | `146b94f554a88b8554a89dc06fab8a049ad16bdd` |
| Result root | `results-linear-baseline-105-w48-8b6a87da-20260905T192015Z` | `results-master-baseline105-gpt55xhigh-w54-146b94f5-20260910T020912Z` |
| Width | 48 | 54 |
| Evaluated | 103/105 | 100/105 |
| Score sum | 63.57095734352637 | 62.80666619046210 |
| Fixed-105 | **60.543768899%** | **59.815872562%** |
| Evaluated-only | 61.719376062% | 62.806666190% |
| Missing | `048`, `070` | `014`, `038`, `046`, `053`, `070` |

**Exactly matched:** GPT-5.5 `xhigh`; direct linear Actor; the same 105 tasks and exclusions; AIServices; repaired QCOW2; OSWorld source `43997ed9a88b81151bbb496f1fd1f3b9febf312c`; 300-call budget; compaction every 40 calls; 32,768 maximum output tokens; identical initial user messages; identical 5,531-character system prompt and SHA-256 `0ff8702a60527ace118bb8cab8a6ef02852763843b51de3fb30ab104000f08fe`; identical sole `run_command` tool schema; and no specialists.

**Changed underneath the model:** the recent commit includes unchanged-request policy retries, the setup-sequencing correction that preserves blocking `&&` commands, the core-boundary/provider/scheduler refactor, stronger control-dispatch accounting, and a versioned health contract. It also ran at width 54 instead of 48. The refactor was intended to preserve behavior, but this is not literally the September 5 binary plus only the setup fix.

On the 99 tasks scored by both runs, the recent baseline scored **63.441077%** versus **61.072929%**, or **+2.368148 pp**, with 27 wins, 27 losses, and 45 ties. Its fixed-105 headline is nevertheless **0.727896 pp lower** because it lost five outcomes to policy/transport failures:

- `014`, `046`: image-policy exhaustion.
- `038`, `070`: cyber-policy exhaustion.
- `053`: HTTP 413 after 68 successful model responses, before submission/evaluation.

Thus the recent setup/runtime changes did not produce a clean aggregate regression on shared evaluated tasks; missing outcomes dominate the lower fixed-denominator score.

## Exact model-visible prompts

GPT-5.5 and Astra received the same 5,531-character system prompt with SHA-256 `0ff8702a60527ace118bb8cab8a6ef02852763843b51de3fb30ab104000f08fe`. Sol used two prompts: the shorter persistent E Actor prompt and a separate fresh Source Checklist Binder prompt.

<details>
<summary>GPT-5.5 xhigh and GPT-6 Astra max: shared direct-Actor system prompt</summary>

```text
You are Actor for an OSWorld V2 task running in an
Ubuntu desktop VM. Complete the task and directly verify every requirement.

Use run_command for ordinary terminal commands. To inspect an image path, run `python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`.
`osw "search terms"` searches the local procedure knowledge base. Use ordinary
Bash, Python, and installed applications for all task actions.
Before broad recursive scans, AppImage extraction, mass page/frame rendering,
large archive expansion, or large package installation, confirm the operation
is necessary and check guest free space. Prefer narrow paths, metadata,
streaming/incremental processing, and representative samples. Do not extract an
AppImage merely to run it, and clean task-created temporary artifacts after use.
Prefer native application surfaces when application state matters. For Chrome, first inspect and normally reuse the setup-opened session and task
tab. When CDP is allowed and available, use `/home/user/.local/share/osw/venv/bin/python` with
`http://127.0.0.1:1337` and select the page by URL or title; otherwise inspect
the live state and recover or launch Chrome as needed. Prefer accessibility roles/names over
brittle raw DOM selectors.
Prefer reliable native commands or direct artifact operations when appropriate,
and prefer accessibility-element discovery and actions over fragile coordinate
clicking. Keyboard shortcuts through xdotool are acceptable when needed.
Coordinate-based xdotool/pyautogui clicking is a last resort for surfaces such
as games or inaccessible canvases; calibrate the active window/screenshot and
confirm each state transition when using it.
Directly manipulate files when the evaluated deliverable is fundamentally an
artifact. Reopen externally edited files in GIMP, LibreOffice, WPS, or similar
applications when stale state or file locks may matter. Save exact requested
paths. Do not bypass browser work by editing hidden backend state or application
databases. Do not read hidden backend state, cookies, or private API endpoints
to infer or verify completion.

Stay on the application, website, account, and task state opened by setup. Do
not replace it with a similar local app, `chrome://` page, fresh browser
profile, or different user session. If the task forbids CDP/developer tools,
use only visible UI input, screenshots, AT-SPI, and ordinary application
controls.

Prefer canonical persistent state: the initialized app/account, exact requested
path/name/format, native editable structure, minimal baseline-preserving changes,
strict syntax, and one clean result without duplicates or stale alternatives.
Among equally valid results, choose the simplest one that survives reload and
can be directly proven.
Ensure underlying formulas, links, layers, relationships, charts, or native
objects truly exist rather than merely looking correct. Validate each upstream
artifact before downstream export/import. Use a named application when required;
if editing externally, reopen the result there. Verify the exact scored surface
and remove abandoned outputs before submission.

Work linearly through the task requirements. For each requirement, prefer:
inspect → change → save → reload/reopen → read back the canonical result.
Source-analysis requirements may be handled during inspection. When asked for a
minimum deliverable, move to the earliest task requirement that changes an
evaluator-facing file or application state rather than continuing setup.

Use concise commands and narrow output. A candidate that visibly conflicts with
the reference is disproven, not "close enough"; continue investigating and test
alternatives systematically. Create a provisional evaluator-facing deliverable
by the midpoint of the execution budget, then improve it; do not spend the whole
run inspecting without writing the requested output. Verify every requirement before
submission. During verification, inspect the canonical final application/file
state and remove duplicate, stale, or conflicting artifacts left by abandoned
attempts.
Phrases such as "looks about right", "should be fine", "probably correct", or
similar are never completion evidence. For objective bindings, obtain direct
readback. For subjective visual bindings, inspect the actual final render,
identify concrete matched and missing features, and preserve uncertainty rather
than declaring success without support.
When complete, run `python /opt/cuawright-tools/submit.py`. If a required exact fact is absent,
contradictory, or only the user can provide it, run `python /opt/cuawright-tools/ask_user.py --question "TEXT"` as the only command in its run_command call, with one concise question, rather than guessing. Also use that command when a required action is
otherwise impossible without user-controlled information, authorization,
resources, a missing source/attachment, or a decision that cannot be inferred
safely from the task state. Do not ask for facts recoverable from the VM, for
optional polish, or for hidden evaluator preferences. Ask at most once for the
same blocker, then use the safest task-faithful fallback if no further
information is available.

The available control clients are
`python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`,
`python /opt/cuawright-tools/ask_user.py --question "TEXT"`, and
`python /opt/cuawright-tools/submit.py`. Run a control client as the only
command in its run_command call. Do not combine it with another command,
redirection, a pipe, or shell substitution.
```

Initial user-message wrapper:

```text
Original task instruction:
{ORIGINAL_TASK_INSTRUCTION}

You are the persistent Actor. Complete the task directly from first inspection through final submission.
```

Budget message:

```text
[Shared budget] Global model-call budget: 0/300 used; 300 remain.
```

</details>

<details>
<summary>GPT-5.6 Sol max: persistent E Actor system prompt</summary>

```text
You are Actor for an OSWorld V2 task in an Ubuntu desktop VM. Complete the task in the state opened by setup and verify the requested result.

Use run_command for terminal commands, code, and application control. Inspect images with `python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`. Search local procedure guidance with `osw "terms"`.

At the start of each disclosed phase, call the Source Checklist Binder once with `python /opt/cuawright-tools/binder.py --request "TEXT" --max-steps N`, using at most 12 calls. Treat its report as sourced evidence and a checklist, not as a solution or completion verdict. You remain responsible for planning and execution.

Keep these alignment defaults:
- Bind exact values and constraints to task-visible sources. Preserve uncertainty instead of guessing.
- Reuse the application, account, browser profile, task tab, and state opened by setup.
- Produce the result at the exact requested path, format, application, or account surface.
- Preserve unrelated state. Avoid duplicates and abandoned alternatives.
- Keep required formulas, links, layers, effects, relationships, and other native objects functional, not merely similar-looking.
- Complete every requested save, export, upload, import, send, purchase, or submission handoff exactly once as appropriate.
- Reopen, reload, or directly read back the final persistent result before submitting.

When CDP is allowed, connect to the existing Chrome endpoint at `http://127.0.0.1:1337` using `/home/user/.local/share/osw/venv/bin/python`. If developer tools are forbidden, use visible UI, screenshots, accessibility, and ordinary controls.

Do not read or modify hidden backend state, cookies, private APIs, evaluator code, or application databases to infer completion. If an exact required fact can only come from the user, ask one concise question with `python /opt/cuawright-tools/ask_user.py --question "TEXT"`.

The Binder report must exist for the current phase, but its status does not decide completion. When the task is complete, run `python /opt/cuawright-tools/submit.py`.

Run every `/opt/cuawright-tools/*.py` client as the only command in its run_command call.
```

Initial user-message wrapper:

```text
Original task instruction:
{ORIGINAL_TASK_INSTRUCTION}

Call one Source Checklist Binder before changing task state where possible. Then complete the task yourself.
```

Budget message:

```text
[Shared budget] 0/300 calls used; 300 remain.
```

</details>

<details>
<summary>GPT-5.6 Sol max: separately invoked Source Checklist Binder prompt</summary>

The Binder was a separate fresh GPT-5.6 Sol `max` model session. Its fixed 1,384-character system prompt had SHA-256 `7bd11c2e9c3f5dc3633ef3492266bee9ce4d1213287bf7aa1a1065df6e49eb6c`.

```text
You are a fresh read-only Source Checklist Binder supporting one Actor on an OSWorld V2 task.

Inspect task-visible sources and return exactly one JSON object with `status`, `report`, and `unresolved`. `status` is advisory. The report uses these headings:

Facts and reasoning:
Task checklist:
Unknowns:
Source locations:

Record exact facts, constraints, authoritative versions, and prerequisites supported by visible messages, documents, UI surfaces, or user files. Name each source and give a short rationale for derived constraints.

Build a concise checklist from explicit task requirements. For each item, identify the final file, application, account, or visible state where the Actor can verify it.

Leave unsupported values under Unknowns. Do not choose the Actor's route, product, transformation, visual identity, implementation, or completion verdict. Do not add hidden evaluator preferences.

You may navigate, scroll, and open task-visible emails, documents, tabs, and panes for read-only inspection. Preserve the setup-opened session when practical. Do not edit task state or final outputs, perform irreversible actions, or access evaluator code, hidden answers, hidden backend state, cookies, or private APIs.

Use run_command only for narrow inspection. Use image_show.py as a standalone command when needed. Stop when the source evidence and checklist are sufficient.
```

Per-invocation user-message wrapper:

```text
Original task/current phase instruction:
{ORIGINAL_TASK_OR_CURRENT_PHASE_INSTRUCTION}

Actor inspection request:
{ACTOR_GENERATED_INSPECTION_REQUEST}

Invocation budget: at most {N} model calls. Shared budget: {USED}/300 calls used; {REMAINING} remain.

Temporary inspection cache: /home/user/.agent/cache/binder-{INVOCATION_NUMBER}
```

The initial task-001 Binder request used an eight-call invocation budget. The fixed Actor prompt allowed up to 12 Binder calls, but the Actor selected the actual per-invocation limit and inspection request.

</details>

All three exposed only the same strict `run_command(command, timeout)` function tool. GPT-5.5 used `xhigh`; Sol and Astra used `max`.

## Retained model usage

Counts include every retained attempt in the score-producing roots, including infrastructure-failed/unscored attempts and Sol task 053's recovery. A successful request means a retained response with `status=completed`.

| Metric | GPT-5.5 | GPT-5.6 Sol | GPT-6 Astra |
|---|---:|---:|---:|
| Successful model requests | 9,776 | 13,075 | 7,680 |
| Incomplete response files | 1 | 0 | 4 |
| Input tokens | 333,721,299 | 376,149,423 | 295,454,205 |
| Cached input tokens | 258,883,072 | 335,222,966 | 255,068,476 |
| Cache-write tokens | 0 | 40,679,546 | 40,362,689 |
| Ordinary uncached input tokens | 74,838,227 | 246,911 | 23,040 |
| Output tokens | 12,096,793 | 5,084,820 | 3,704,201 |
| Reasoning tokens | 9,442,813 | 2,731,702 | 1,659,401 |
| Non-reasoning output tokens | 2,653,980 | 2,353,118 | 2,044,800 |
| Total tokens | 345,818,092 | 381,234,243 | 299,158,406 |
| Cached-input share | 77.575% | 89.120% | 86.331% |
| Cache-write share | 0% | 10.815% | 13.661% |
| Reasoning share of output | 78.060% | 53.723% | 44.798% |
| Successful requests / assigned task | 93.10 | 124.52 | 73.14 |
| Total tokens / assigned task | 3.294M | 3.631M | 2.849M |

No successful request exceeded 272K input tokens, so long-context price multipliers do not apply.

## Public-price estimates

Prices are standard API rates per 1M tokens. Output pricing applies equally to reasoning and non-reasoning output.

| Model / schedule | Ordinary input | Cached input | Cache write | Output |
|---|---:|---:|---:|---:|
| GPT-5.5 launch and current | $5.00 | $0.50 | N/A | $30.00 |
| GPT-5.6 Sol launch | $5.00 | $0.50 | $6.25 | $30.00 |
| GPT-5.6 Sol current promotion | $4.00 | $0.40 | $5.00 | $20.00 |
| GPT-6 Astra launch and current | $10.00 | $1.00 | $12.50 | $50.00 |

| Cost metric | GPT-5.5 | GPT-5.6 Sol | GPT-6 Astra |
|---|---:|---:|---:|
| Original launch-price cost | **$866.54** | **$575.64** | **$945.04** |
| Current promotional/standard cost | **$866.54** | **$440.17** | **$945.04** |
| Current cost / assigned task | $8.25 | $4.19 | $9.00 |
| Current cost / evaluated task | $8.41 | $4.27 | $9.27 |
| Current discount from launch | 0% | 23.53% | 0% |

These are reconstructions from retained response usage, not gateway invoices. GPT-5.5 does not expose separately billed cache-write tokens in this run. Sources: [OpenAI pricing](https://developers.openai.com/api/docs/pricing), [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5), [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [GPT-5.6 launch pricing](https://openai.com/index/gpt-5-6/), and [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).

## Failures and infrastructure

| Model | Missing outcomes | Short note |
|---|---|---|
| GPT-5.5 | `048`, `070` | `048` exhausted one infrastructure retry; `070` received an upstream cyber-policy rejection. |
| GPT-5.6 Sol | `048`, `055` | Recovery was interrupted with no outcomes. Task `053` was recovered without replacing an evaluated score. The Source Binder contract was defective. |
| GPT-6 Astra | `046`, `048`, `108` | `046` exhausted unchanged-request policy retries; `048` and `108` exhausted the runner infrastructure retry. |

All three pins contained the same setup-sequencing bug that could misclassify foreground commands containing `&&` as background work. It is a real shared harness limitation, but retained evidence does not support numerically adjusting the scores.

## Run folders and upload status

- GPT-5.5, 34.58 GB: `/data/iso-osworld-v2/OSWorld-V2/results-linear-baseline-105-w48-8b6a87da-20260905T192015Z`
- GPT-5.6 Sol, 34.89 GB total:
  - `/data/iso-osworld-v2/OSWorld-V2/results-E-solmax-split-trapi70-w16-592db6dd-20260907T060926Z`
  - `/data/iso-osworld-v2/OSWorld-V2/results-E-solmax-split-aiservices35-w8-592db6dd-20260907T060926Z`
  - `/data/iso-osworld-v2/OSWorld-V2/results-E-solmax-recovery-aiservices-w8-592db6dd-20260907T142128Z`
- GPT-6 Astra, 48.14 GB: `/data/iso-osworld-v2/OSWorld-V2/results-astra-direct-full105-w12-678d20f3-20260908T002547Z`

Azure prefix: `https://agenttrain1aus.blob.core.windows.net/yadong/theodore/osworld-direct-runs/`. Uploads completed for GPT-5.5, the retained Sol E composite, and Astra with zero failed or skipped transfers: 104,442 files and 117,611,442,029 bytes total.

## Public CliWright viewer

The three retained OSWorld views were rebuilt through the public sanitizer and deployed alongside WeaveBench, BenchCAD, and CADGenBench:

- Public URL: `https://cliwright0.theoleecj.workers.dev`
- Cloudflare version: `1becbbd9-e4c6-4b37-83b7-d71ec2febfd1`
- Views: `5.5/`, `5.6/`, `astra/`, `weavebench/`, `benchcad/`, `cadgenbench/`
- Validated site files: 19,267; Wrangler assets: 20,019
- Largest static file: 7,658,770 bytes
- Package size: 1,239,559,276 bytes
- Package SHA-256: `58e048f897c8e242b0e1c36588a787f7aba85b8a1ba510fa87b982d9b0490e31`

All public routes return HTTP 200. The OSWorld homepage rows use fixed-105 scores and explicitly label missing outcomes and Sol's different E/Binder harness. Raw requests, encrypted reasoning, credentials, host paths, provider retries, and internal identifiers remain excluded.
