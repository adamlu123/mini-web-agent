---
name: terminal-task-run
description: Run RST (Recursive Synthetic Terminal Tasks) trajectory generation through the Docker harness on the phyagi gateway — preparing the task pool, launching a batch band by band from easy to hard, monitoring, scoring against each task's private verifier, and cleaning up. Use when asked to run, launch, monitor or score terminal (non-web) task generation, or when working with terminal_docker, terminal_rst.yaml, or run.benchmarks.rst.
---

# RST terminal task runs (phyagi)

Generates agent trajectories on **RST** tasks — command-line tasks that ship their
own Docker environment and their own private verifier. This is the terminal
counterpart to the web `om2w` flow; it shares the agent loop, compaction and
self-reflection, and nothing else.

The working set is **5,000 tasks in 10 bands of 500**, ordered easy → hard by the
length of the reference solution (the paper's own difficulty proxy). Run them band
by band: `g01` first, and only move up once the pass rate at the current band is
understood.

| band | `solve.sh` non-empty lines | band | lines |
|---|---|---|---|
| g01 | 11 – 51 | g06 | 115 – 130 |
| g02 | 51 – 68 | g07 | 130 – 145 |
| g03 | 68 – 83 | g08 | 145 – 160 |
| g04 | 83 – 99 | g09 | 160 – 175 |
| g05 | 99 – 115 | g10 | 175 – 189 |

Each task package:

```
<task_id>/
  instruction.md            # what the agent is shown
  task.toml                 # difficulty, timeouts, cpu/mem
  environment/Dockerfile    # built into the container the agent lives in
  solution/solve.sh         # PRIVATE — never mounted
  tests/test.sh             # PRIVATE — run only after the episode ends
  tests/test_state.py
```

## How it differs from the web flow

- **No browser, no screenshots, no Browserbase, no WebJudge.** Nothing to release
  after a kill; the cleanup target is Docker, not Browserbase sessions.
- **The agent runs inside the task's own container** (`terminal_docker`), not on the
  host. `/tests` and `/solution` are absent from that container.
- **Two verdicts, kept separate.** Self-reflection is the agent's *own* completion
  gate (`judge_mode: trajectory`, same as the web `persistent_cli` config, over a
  `terminal-steps.jsonl` manifest the environment appends per command). The task's
  `tests/test.sh` is the *external* score, run post-hoc, and the agent never sees it.
- **`self_reflection` is a real command inside the container**, mounted read-only
  from `environments/container_bin/self_reflection`. It is a POSIX-sh client: it
  writes its argv under `/harness/.reflect/` and a watcher thread on the host runs the
  actual judge and writes the result back. Credentials stay on the host; no python
  is needed in the task image; the image is not rebuilt.
- `agents/default.py` and `tools/self_reflection.py` are shared with the web flow.
  Do not fork them for terminal changes.

## Environments

**Target (what this skill assumes):**

| | |
|---|---|
| host | Linux, native `x86_64` |
| policy | phyagi gateway, `gpt-5.4` (`model_class: phyagi`) |
| judge | phyagi gateway `/responses` (`generation/judge_phyagi.yaml`) |
| credentials | `source /home/luyadong/cred.sh` → `OPENAI_GATEWAY_API_KEY` |
| python | `/home/luyadong/.venv/bin/python` |
| docker | daemon reachable by the run user; task images are `linux/amd64`, native here |

**Verified on (where this harness was built and smoke-tested):**

| | |
|---|---|
| host | macOS 15 / Apple M2 Pro, `arm64` |
| docker | OrbStack 29.4.0, 12 CPU / 8 GB VM, `linux/amd64` under Rosetta |
| policy | Azure OpenAI `gpt-5.4` (`generation/model_azure_gpt54.yaml`) |
| judge | Azure OpenAI chat-completions (`generation/judge_azure_gpt54.yaml`) |

What that means for the target host:

- `platform: linux/amd64` in the config is a no-op on Linux/amd64 and a Rosetta
  emulation request on Apple silicon. Leave it set; it keeps both hosts identical.
- Timings below were measured under emulation and include per-task image builds.
  Native Linux is faster on the container side; the agent loop is gateway-bound and
  will not change much.
- **The phyagi judge route has not been smoke-tested end to end** — only the Azure
  route was. The two backends need opposite argv shapes (see Known limits), and the
  phyagi shape is covered by a unit check but not a live call. Confirm it on the
  first `--limit 2` run before launching a band.

## Prepare the task pool

Once per machine. Downloads ~3.9 GB, extracts ~600 MB of task packages, and writes
the band manifests.

```bash
source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python \
  .claude/skills/terminal-task-run/scripts/prepare_rst_tasks.py \
  --dest /home/luyadong/data/rst
```

It downloads `Zhongzhi1228/Recursive-Task-Synthesis`, collapses near-duplicate
rewrite variants to one representative per cluster (37,484 → 12,010; the paper keeps
variants undeduplicated and says so), takes the shortest 5,000, splits them into 10
bands, extracts the packages, and **drops the ~583 multi-container tasks** the
harness cannot run. Selection is deterministic — the same inputs give the same 5,000.

Output:

```
/home/luyadong/data/rst/selection/
  tasks/tasks/<task_id>/     <- tasks_root is this directory
  g01.jsonl .. g10.jsonl     <- ~440 tasks each after the compose filter
  all.jsonl
```

Needs `huggingface_hub`, `pyarrow`, `pandas` in the venv.

## Launch one band

Same tmux convention as the web runs: **do not use `screen`**; open a pane in the
`inference` window of the current session.

```bash
BAND=g01
DATA=/home/luyadong/data/rst/selection
OUT=/home/luyadong/sandbox/mini-web-agent/outputs/terminal/rst_${BAND}

mkdir -p "$OUT"
tmux has-session 2>/dev/null || { echo "Not in a tmux session"; exit 1; }
tmux list-windows -F '#W' | grep -qx inference || tmux new-window -n inference
tmux split-window -t inference -v \
  "source /home/luyadong/cred.sh && \
   cd /home/luyadong/sandbox/mini-web-agent && \
   /home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.rst \
     -c generation/terminal_rst.yaml \
     -c generation/judge_phyagi.yaml \
     -c environment.tasks_root=$DATA/tasks/tasks \
     -c environment.build_timeout_seconds=1200 \
     -c agent.step_limit=100 \
     --tasks-file $DATA/${BAND}.jsonl \
     --workers 8 \
     --output-dir $OUT 2>&1 | tee $OUT/run.log; \
   exec bash"
tmux select-layout -t inference tiled
```

**Always smoke-test first**: add `--limit 2 --workers 2` and confirm two tasks reach
`summary.json` with a non-null `score` before launching the full band.

Two flags are not optional:

- **`-c generation/judge_phyagi.yaml`** — without an explicit judge endpoint,
  `self_reflection` falls back to its TRAPI Kimi default and needs
  `az login --scope api://trapi`. Batch hosts do not have that: every reflection call
  fails with a `ChainedTokenCredential` error while the run keeps burning steps.
- **`-c environment.tasks_root=...`** — the default in `terminal_rst.yaml` is a local
  dev path that does not exist on the batch host.

### Config layering

Later `-c` specs win, so overlays compose:

| spec | purpose |
|---|---|
| `generation/terminal_rst.yaml` | base: prompts, `terminal_docker`, phyagi policy |
| `generation/judge_phyagi.yaml` | judge → phyagi `/responses` |
| `generation/model_azure_gpt54.yaml` | policy → Azure (local dev only) |
| `generation/judge_azure_gpt54.yaml` | judge → Azure (local dev only) |

### Useful overrides

- `--limit N` — first N tasks of the band.
- `--group N` — filter by the `group` field, for running from `all.jsonl`.
- `-c agent.step_limit=100` — matches `persistent_cli`; 60 is too tight, tasks finish
  the work and then run out of steps inside the reflection loop.
- `-c agent.require_self_reflection_success=false -c agent.step_limit=50` — disable
  the completion gate. On 20 `g01` tasks this produced the same 11 positives as the
  gated run at 285 s per positive instead of 508 (the gate lost 1–3 correct runs to
  the step limit and passed 4 wrong ones). Labels come from `tests/test.sh` either
  way, so for rejection sampling this is the cheaper setting.
- `-c environment.reuse_images=false` — force rebuilds after editing a Dockerfile.
- `-c environment.command_timeout_seconds=...` — per-command ceiling, default 240 s.

### Choosing `--workers`

Each container asks for ~2 GB (`task.toml`). Size workers against host RAM, not
CPU count, and leave headroom for the image builds running alongside.

## Monitor

```bash
tail -f $OUT/run.log
find $OUT -name summary.json | wc -l              # tasks finished
docker ps --filter name=rst- --format '{{.Names}} {{.Status}}'
```

Per-task artifacts land in `$OUT/<task_id>/`:

| file | contents |
|---|---|
| `trajectory.json` | full message history + cumulative token usage — **this is the SFT data** |
| `summary.json` | steps, exit_status, score, partial_credit, seconds |
| `verifier_log.txt` | raw output of the private `tests/test.sh` |
| `verifier_result.json` | `{score, returncode, tests_passed, partial_credit}` |
| `plan.md`, `judge_config.json`, `final_runs/run_<id>/` | what the agent authored |
| `steps/`, `logs/` | one file per executed command and its output |
| `docker_build.log` | image build output; only worth reading on failure |

`batch_summary.json` at the root has the roll-up.

## Score

`score` comes from the task's own verifier, not from a judge model: `1` = every
assertion passed, `0` = at least one failed, `null` = the verifier produced no
reward — treat that as infrastructure failure, not as a wrong answer.

```bash
python - <<'PY'
import json, pathlib, collections
root = pathlib.Path("OUT_DIR_HERE")
rows = [json.loads(p.read_text()) for p in root.glob("*/summary.json")]
scored = [r for r in rows if isinstance(r.get("score"), int)]
print(f"{sum(r['score'] for r in scored)}/{len(scored)} passed"
      f"   raw={dict(collections.Counter(str(r.get('score')) for r in rows))}")
print("exit:", collections.Counter(r.get("exit_status") for r in rows))
print("skipped:", sum(1 for r in rows if r.get("skipped")),
      " errors:", sum(1 for r in rows if r.get("error")))
PY
```

A task carrying `skipped` was never run (see Known limits) and is not part of the
pass rate.

`exit_status` reads:

- `Submitted` — the agent passed its own reflection gate and declared done.
- `LimitsExceeded` — hit `step_limit`. **This does not mean the task failed**; a run
  can do the work correctly and still run out of steps inside the reflection loop.
  Check `score` separately.

For rejection sampling, keep trajectories where `score == 1`. A `Submitted` run with
`score == 0` is a self-reflection false positive and must not enter the SFT set.

## Kill and clean up

Send `C-c` to the run pane, or kill the pane. Both deliver the signal to the whole
foreground process group, which is what you want:

```bash
tmux send-keys -t inference.<pane_index> C-c
# or
tmux kill-pane -t inference.<pane_index>
```

Avoid `pkill -f 'benchmarks[.]rst'` on its own. The worker processes are spawned
with a `multiprocessing.spawn` command line that does not contain that string, so
it kills only the parent; workers mid-episode keep their containers and gateway
calls going and then drain the items the parent had already queued. If you must
use pkill, kill the process group instead:

```bash
kill -- -"$(pgrep -f 'benchmarks[.]rst' | head -1 | xargs ps -o pgid= -p | tr -d ' ')"
```

Workers also watch their parent pid and exit on their own within a few seconds of
it disappearing, so a stray orphan is a bug worth reporting, not an expected state.

The environment removes its container on normal close; a killed run leaks them.

```bash
docker ps -a --filter name=rst- -q | xargs -r docker rm -f
```

**Only remove `rst-`-prefixed containers.** The daemon is shared; a blanket
`docker rm -f $(docker ps -aq)` will kill colleagues' work.

Images are cached by a content hash of `environment/`, so a full band leaves several
hundred images. Reclaim with `docker image prune -a` only when no run is active — a
rebuild costs roughly two minutes per task.

## Cost and duration

Measured with `gpt-5.4` on two `g01` tasks, on the local dev host (Azure policy,
Apple-silicon emulation, cold image cache):

| task | steps | wall | billed input | cached input | output |
|---|---:|---:|---:|---:|---:|
| pandas dropna → csv | 19 | 134 s | 124,661 | 90,368 | 5,134 |
| openssl RSA/AES | 26 | 323 s | 66,236 | 233,728 | 19,051 |

Roughly **230 s, 95 k billed input, 12 k output per task** at `g01`, with a first-time
image build worth about 130 s of that.

Input dominates and grows quadratically with step count: every step resends the whole
history, so 26 steps accumulates 300 k input tokens. **Cost tracks step count and
cache hit rate, not output length.** Cache hit rates were 42% and 78% on these two.

Treat these as a floor for the higher bands: `g10` tasks have reference solutions
3–4× longer than `g01` and will use more steps.

## When a batch is slow

Throughput has three independent causes that need opposite fixes, so measure before
changing anything:

```bash
python .claude/skills/terminal-task-run/scripts/diagnose_throughput.py \
  --output-dir $OUT --workers <N> --step-limit 80
```

Read it in this order:

| what the report shows | cause | fix |
|---|---|---|
| `FormatErrorRetry` > 0 | the prompt's output shape and `model.response_mode` disagree; every retry is a wasted round trip and is **invisible in `n_steps`** | align `model.response_mode` with `agent.system_template` (`sft_state` for `<think>/<bash>` tags, `json_schema` for a JSON object) |
| build is a large share of task time, few cache hits | image builds | raise `--workers`, lower `--build-workers`; builds are host-bound, episodes are gateway-bound |
| `at step_limit` high, `self_reflect calls` ≥ 3 | episodes not converging; the reflection gate keeps rejecting | see below |
| `sec/step` ≥ 30 with normal step counts | gateway queueing | **lower** concurrency; more workers only lengthens the queue |
| real wall clock ≫ `ideal wall at Nw` | workers starved | check the disk line: >85% used makes the daemon slow at both build and exec |
| docker disk climbing run over run | images or build cache leaking | `remove_image_on_close: true` (default in this config) and `--prune-every 50` |

**A quick sanity number**: worker-seconds per task is `wall_clock × workers ÷ tasks`.
At `g01` a healthy value is 250–400 s. Much above that, something in the table applies.

**If it is reflection churn**: the gate exists because the web harness has no ground
truth. Here `tests/test.sh` is ground truth, so the gate does not improve label
quality — it only helps the agent self-correct, at the cost of a judge round trip
plus the steps spent reacting to each rejection. Turning it off is a real option:

```bash
-c agent.require_self_reflection_success=false -c agent.step_limit=40
```

That trades yield (fewer positives per task) for cost (far fewer steps per task).
Which wins is empirical: A/B a band and compare **seconds per accepted trajectory**,
not pass rate.

## Prompts

All agent-facing prompt text lives in
**`src/miniswewebagent/config/generation/terminal_rst.yaml`**:

- `agent.system_template` — the main prompt. Contains the JSON response contract, the
  task-workspace / harness-workspace split, the workflow, the verification contract
  (the assertion categories the agent's own `verify_state.py` must cover), and the
  self-reflection spec.
- `agent.instance_template` — per-task injection: instruction, paths, done checklist.
- `agent.summary_user_prompt` — compaction. Overridden here because the default in
  `agents/default.py` is written for the web harness and talks about screenshots and
  selectors.
- `model.observation_template`, `model.format_error_template`.

**The self-reflection rubric is not a file.** `judge_config.json` is authored by the
agent at runtime, constrained by the `## Self-reflection` section of
`system_template`. To change judge behaviour, edit that section.

## Known limits

- **Multi-container tasks are skipped, not run.** 583 of the 5,000 ship a compose
  file defining several networked containers (an SSH target, a mock API server, a
  private subnet with fixed IPs), and the instruction depends on that network
  existing. `terminal_docker` builds one image and runs one container, so those
  tasks would fail for environment reasons and look like model failures.
  `prepare_rst_tasks.py` filters them out of the band manifests; if one reaches the
  runner anyway it is recorded as `skipped` in `summary.json`, counted separately in
  `batch_summary.json`, and left out of the pass-rate denominator. They are spread
  evenly across bands (51–64 per band), leaving ~440 runnable per band.
- **Self-reflection is wrong in both directions.** It has passed a run whose output
  had the wrong column headers — the agent's own verifier only checked that headers
  existed, not that they matched the values discoverable in the workspace — and it
  has blocked runs that were already correct. Never treat `Submitted` as a score.
- **The phyagi judge route is not smoke-tested.** `self_reflection` needs opposite
  argv shapes per backend: a `/responses` gateway takes the real deployment name,
  while an OpenAI chat-completions server needs the `policy` sentinel (passing a chat
  URL with a real model name silently routes to the responses backend). The
  environment infers this from the endpoint; `judge_backend: responses|policy_chat`
  forces it. Verify on the first `--limit 2` run that
  `final_runs/run_*/judge_result.json` exists and has `predicted_label` set.
- **`/logs/verifier` is created by the harness** before running `test.sh`. 4,960 of
  the 5,000 RST tasks create it themselves, so a regression here would show up only
  on the remaining 40 — as `verifier_log.txt` ending in `No such file or directory`
  while pytest was green, and `score: null`.
