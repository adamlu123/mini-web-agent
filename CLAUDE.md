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
CFG=archive/best_default_judge_json_openrouter_qwen35_9b.yaml

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
- The config `archive/best_default_judge_json_openrouter_qwen35_9b.yaml` uses `model_class: openrouter`, `model_name: qwen/qwen3.5-9b`. Confirmed valid OpenRouter slug as of 2026-05-21.
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

Config specs passed to `-c` are resolved as `src/miniswewebagent/config/<spec>`, so
subdirectories must be included in the spec (e.g. `-c generation/…`).

- `generation/best_default_judge_json.yaml` — default phyagi gpt-5.4 (gateway).
- `generation/best_default_judge_json_agnostic.yaml` — browser-backend-agnostic variant of the above.
- `generation/best_default_judge_json_persistent_cli.yaml` — persistent incremental browser variant.
- `archive/best_default_judge_json_openrouter_qwen35_9b.yaml` — OpenRouter Qwen3.5-9B, plain agent (no oracle CLI).
- `archive/best_default_judge_json_openrouter_qwen36_cli.yaml` — OpenRouter Qwen3.5-9B with per-task oracle CLI tool injected (`outputs/cli/0426_oracle_cli`). Higher success rate, not a true baseline. (Note: filename says `qwen36_cli` but the configured slug is `qwen/qwen3.5-9b`.)
- `archive/best_default_judge_json_kimi_openrouter.yaml` — OpenRouter Kimi variant.

## vLLM eval: Qwen3.5 4B/9B SFT/RL checkpoints (SPB)

Evaluation only — no training code lives here. Serve a converted HF checkpoint with
vLLM, then point the normal `om2w` batch runner at it.

```bash
# 1. serve the checkpoint (any OpenAI-compatible server works)
vllm serve /path/to/ctx2-<variant>-hf-vlm --tp 8 --max-model-len 32768

# 2. run the benchmark against it
python -m miniswewebagent.run.benchmarks.om2w \
  -c eval/om2w_spb_vllm_sw10.yaml \
  -c model.endpoint=http://127.0.0.1:8000/v1 \
  -c model.model_name=sft_ckpt \
  --tasks-file src/miniswewebagent/run/benchmarks/om2w_260220.json \
  --output-dir "$OUT"
```

Configs (`src/miniswewebagent/config/eval/`) differ only in how history is fed to the
model; all use `model_class: openai_compatible`, `response_mode: sft_state`,
`output_truncation_chars: 24000`, `step_limit: 50`, no compaction:

| config | context construction |
|---|---|
| `eval/om2w_spb_vllm_full24k.yaml` | full history |
| `eval/om2w_spb_vllm_sw10.yaml` | `context_window_steps: 10` (sliding window) |
| `eval/om2w_spb_vllm_lastobs.yaml` | `history_context_mode: last_obs` — history obs stubbed, latest full |
| `eval/om2w_spb_vllm_lastobs_think.yaml` | `history_context_mode: last_obs_think` |

`model_class` accepts `vllm` or `openai_compatible` (same class). The endpoint is
normalized, so `:8000`, `/v1`, and `/v1/chat/completions` all work; the API key
defaults to `dummy`. Env fallbacks: `OPENAI_COMPATIBLE_{ENDPOINT,MODEL,API_KEY}`.

Notes / limitations:
- **`last_obs_think` currently behaves the same as `last_obs`.** The think-only trim
  rewrites message *content*, but under `sft_state` the model layer rebuilds assistant
  turns from `extra.raw_response` (`_sft_state_assistant_content`), restoring `<bash>`.
  This matches upstream `rl_yifei_clean`, which is where the published SPB scores came
  from — do not "fix" it without re-baselining.
- The `sum10` (compaction) variant was **not** ported: it needs
  `summary_max_output_tokens` / `summary_response_mode` / a `summary_text` parse mode
  plus a compaction wrapper kept byte-aligned with the training bundle, and it has no
  validated score upstream.
- Model-layer `max_context_tokens` / `sliding_window_keep_turns` token eviction was
  **not** ported: the upstream standard command sets `28672` + `999`, which disables it
  ("超窗即停" — an over-window request is sent as-is, vLLM 400s, the episode ends).
  Context control lives at the agent level instead.

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

## Changelog

### 2026-07-29 — vLLM inference/eval for Qwen3.5 4B/9B (ported from `rl_yifei_clean`)

Minimal eval-only port of the SPB pipeline (`docs/PIPELINE_SPB.md` on that branch).
No trainer, no cluster submission scripts, no new top-level folders.

- **New** `src/miniswewebagent/models/openai_compatible_model.py` — OpenAI-compatible
  backend for local vLLM servers. Subclasses `OpenRouterModel` and overrides only auth:
  endpoint normalization, `dummy` API key, defaults of `response_mode: sft_state` /
  text-only / 4096 output tokens.
- **New** `src/miniswewebagent/config/eval/om2w_spb_vllm_{full24k,sw10,lastobs,lastobs_think}.yaml`
  — taken from the branch, repointed at this repo (dropped cluster `judge_python` /
  `judge_script` so the defaults find the vendored `om2w_judge/run.py`, per-variant
  `output_dir`, and `PYTHONPATH` pointed at `agent_runtime/`).
- `models/__init__.py` — registered `vllm` and `openai_compatible`.
- `models/phyagi_model.py` — added `parse_sft_state_output()` for the
  `<think>/<bash>/<done>/<final_response>` format (plus a prefill-style fallback for
  checkpoints that omit the opening `<think>`), dispatched ahead of the existing modes.
- `models/openrouter_model.py` — under `sft_state`, past assistant turns are replayed to
  the model in the tag format they were trained on, rebuilt from `extra.raw_response`
  (the harness otherwise stores only the parsed thought as content). Other response
  modes are untouched.
- `agents/default.py` — added `context_window_steps` (keep last N assistant turns,
  merging the task with the window-start user block so no two user turns are adjacent)
  and `history_context_mode` (`last_obs`, `last_obs_think`). Both default to off, so
  existing configs are unaffected.

Verified: parser accepts valid action/done/prefill outputs and rejects the malformed
cases; both context transforms produce the documented shapes and are non-mutating; all
four configs load through the real loader and build a model with `-c` overrides applied;
a round trip against a fake vLLM server confirms auth headers, payload, SFT-format
replay, action parsing, and token accounting. Test suite was 15 failed / 74 passed both
before and after — identical, no regressions (failures are missing-browser env issues).

Two config bugs that only surfaced on a live run (fixed here):

- `prepend_tools_to_pythonpath` / `prefer_current_python` came over from the branch but
  are **not read** by this repo's `LocalWorkspaceEnvironment`. Without them the agent's
  `python -m browser_session` failed on every task with `python: command not found`.
  Replaced with an explicit `PYTHONPATH: …/agent_runtime` (the `config/generation/*.yaml`
  convention); the launcher must also put a venv with a `python` binary on `PATH`.
- `judge_endpoint: ""` makes `om2w.py` fall back to **direct OpenAI** rather than the
  gateway (`_resolve_judge_api_key` only consults `OPENAI_GATEWAY_API_KEY` when a
  gateway URI is set). Restored `http://gateway.phyagi.net/api/responses`.

Runtime notes for this host: the runner venv needs `openai`, `pillow`, and `backoff`
beyond `pyproject.toml` (`openai` is imported by `phyagi_model.py` but is not declared as
a dependency). Serving Qwen3.5 via the shared `phitrain` venv needs
`FLASHINFER_DISABLE_VERSION_CHECK=1` and `VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer's
sampling kernel does not compile against the installed CUB.
