# CLAUDE.md — web-agent RL run commands

Quick reference for launching, monitoring, and inspecting runs in this worktree.
For prose / design background see [`docs/web_agent_training.md`](docs/web_agent_training.md) and the [README](README.md).

---

## Layout

```
/home/luyadong/sandbox/echo-rl/.claude/worktrees/web-agent-rl/   <-- this worktree (echo-rl code)
/home/luyadong/sandbox/SkyRL/                                    <-- SkyRL repo, target of apply_hooks
/home/luyadong/sandbox/SkyRL/echo_rl/web_agent/                  <-- copy of this worktree's echo_rl/, used at runtime
/home/luyadong/sandbox/SkyRL/echo_configs/                       <-- copy of this worktree's configs/, used at runtime
/home/luyadong/sandbox/mini-web-agent/                           <-- om2w_judge lives here; MINI_WEB_AGENT_ROOT points here
```

The training entrypoint runs inside SkyRL's repo. `apply_hooks_to_skyrl.sh` copies `echo_rl/` and `configs/` into SkyRL, but plain `cp` works too for incremental edits.

## Environment

- Conda env: `echo-rl` (python 3.12)
- Critical pinned versions: torch 2.8.0 (cu128), flash-attn 2.8.1 (cu12torch2.8 wheel), transformers 5.3.0, vllm 0.19.0, ray 2.51.1, tilelang 0.1.9
- Credential files (sourced by the launcher):
  - `/home/luyadong/sandbox/cred.sh` — BROWSERBASE_*, HF_TOKEN, OPENAI_GATEWAY_ENDPOINT (and a stale OPENAI_GATEWAY_API_KEY that gets `unset`)
  - `/home/luyadong/cred_gateway.sh` — working OPENAI_API_KEY for the phyagi gateway

---

## Launch full training (4 GPU, default mode)

```bash
bash scripts/run_web_agent_hard_4gpu.sh
```

This wrapper sources both cred files, activates the conda env, exports `MINI_WEB_AGENT_ROOT` / `ECHO_RL_DATA` / `OUTPUT_DIR`, `cd`'s into SkyRL, and runs `python -m echo_rl.web_agent.entrypoint`.

Defaults:
- `CONFIG_NAME=qwen35_4b_web_agent_hard_4gpu.yaml`
- `OUTPUT_DIR=${WORKTREE}/outputs/qwen35_4b_hard_4gpu`
- `ECHO_RL_DATA=${WORKTREE}/data/web_agent`
- `SKYRL_DIR=/home/luyadong/sandbox/SkyRL`

Override:

```bash
# Self-launch (agent orchestrates Browserbase sessions) mode
CONFIG_NAME=qwen35_4b_web_agent_hard_4gpu_self_launch.yaml \
  bash scripts/run_web_agent_hard_4gpu.sh

# Different output dir
OUTPUT_DIR=/tmp/my_run bash scripts/run_web_agent_hard_4gpu.sh
```

Background launch + tee:

```bash
bash scripts/run_web_agent_hard_4gpu.sh > /tmp/web_agent_training.log 2>&1 &
tail -F /tmp/web_agent_training.log | grep --line-buffered -E "Finished: 'step'|reward/avg_raw_reward|^Traceback"
```

### Resume vs fresh start

`resume_mode: latest` (SkyRL default) — the trainer will pick up the latest checkpoint in `${OUTPUT_DIR}/ckpts/global_step_*`. To force a fresh step-0 start, wipe ckpts first:

```bash
rm -rf "${OUTPUT_DIR}/ckpts"/*
```

Checkpoint retention: `max_ckpts_to_keep: 2` in the YAML, so only the two most recent are kept (~32 GB for Qwen3.5-4B FSDP shards).

---

## Prompt modes

The system + user prompt sent to the policy is built by `format_web_task_prompt(task, start_url, parser_name, tokenizer, add_instruction_prefix=True, mode="default")` in `echo_rl/web_agent/prompts.py`.

Two modes ship today (`list_modes()` returns them):

| `mode` | What the policy is told | When used |
| --- | --- | --- |
| `"default"` | One Playwright tab pre-injected as `page`/`context`/`browser` into every `python -c '...'` snippet. No browser orchestration to do. Documents only `python -c` patterns. | `qwen35_4b_web_agent_hard_4gpu.yaml` (the main training config), and `run_real_rollout.py` (which always passes `mode="default"` today). |
| `"self_launch_persistent_browser"` | Same qwen35 tool-call format, but the agent orchestrates Browserbase sessions itself: one persistent session (descriptor at `${WEB_AGENT_WORKSPACE}/.persistent_session.json`) plus on-demand exploration sessions, and self-judges via `image_qa` / `self_reflection` CLIs. | `qwen35_4b_web_agent_hard_4gpu_self_launch.yaml`. |

**During training** the dataset reads `generator.prompt_mode` from the YAML and threads it through to `WebAgentTaskDataset(prompt_mode=...)` → `format_web_task_prompt(..., mode=...)`. To switch modes for a training run, edit `prompt_mode:` in the YAML (or pick the matching config).

**For one-off rollouts via `run_real_rollout.py`**, the script does **not** currently surface a `--prompt-mode` flag — it always uses `mode="default"`. To inspect the self-launch prompt end-to-end you can either (a) add a one-liner flag and re-invoke, or (b) construct the messages in a Python shell:

```python
from transformers import AutoTokenizer
from echo_rl.web_agent.prompts import format_web_task_prompt

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
msgs = format_web_task_prompt(
    task="Find a 2022 Tesla Model 3 on CarMax.",
    start_url="https://www.carmax.com",
    parser_name="qwen35",
    tokenizer=tok,
    mode="self_launch_persistent_browser",
)
print(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
```

## Real one-task rollout (no SkyRL, no training)

Loads Qwen3.5-4B with HF transformers, drives a real Browserbase session, scores with the o4-mini OSW judge. Saves a full transcript JSON. Best signal-per-minute for "what does the policy look like right now".

```bash
source /home/luyadong/miniconda3/etc/profile.d/conda.sh && conda activate echo-rl
source /home/luyadong/sandbox/cred.sh
source /home/luyadong/cred_gateway.sh
unset OPENAI_GATEWAY_API_KEY

python -m echo_rl.web_agent.scripts.run_real_rollout \
    --task "Find a 2022 Tesla Model 3 on CarMax." \
    --task-id carmax-tesla-001 \
    --start-url https://www.carmax.com \
    --max-turns 8 --max-new-tokens 768
```

Outputs:
- Transcript JSON: `/tmp/web_agent_real_rollout/transcript_<task-id>.json`
- Per-turn screenshots: `/tmp/web_agent_real_rollout/<task-id>_<ts>/screenshots/`

The judge call needs `pip install backoff` in the `echo-rl` env (it's not part of the default install, only training-time Ray workers pick it up via another dep).

---

## Stub smoke test (no GPU / no network)

```bash
pytest tests/test_web_agent_e2e.py
```

In-memory stub browser + deterministic substring reward. Useful sanity-check after editing the env or prompts.

---

## Data preparation

Build hard / easy parquets from the om2w JSON:

```bash
python -m echo_rl.web_agent.scripts.prepare_om2w_data \
    --input /home/luyadong/sandbox/nano_eval/task_files/om2w_260220.json \
    --output data/web_agent/om2w_easy_train.parquet \
    --val-output data/web_agent/om2w_easy_val.parquet \
    --levels easy --train-size 64 --val-size 12 --seed 7
```

Replace `easy` with `medium` or `hard`, adjust `--train-size` / `--val-size`. Available pools: easy=80, medium=143, hard=77.

---

## Inspecting rollouts

After a training run (or while one is running) the per-rollout workspaces are at:

```
/tmp/web_agent_rollouts/<task_id>_<ts>/screenshots/        # PNG per turn
/tmp/web_agent_rollouts/<task_id>_<ts>/.persistent_session.json   # Browserbase descriptor
```

Cleanup is manual: `rm -rf /tmp/web_agent_rollouts/`. Each rollout's Browserbase session is auto-released on cleanup (see `WebAgentEnvironment._release_rollout_browserbase_sessions`).

The model's **response text is not persisted by default**. Two ways to see it:
- One-off: use `run_real_rollout` (above) → transcript JSON.
- During training: set `trainer.dump_data_batch: true` in the YAML → pickled batches written to `${OUTPUT_DIR}/skyrl_logs/data_batches/`. Decode with `tokenizer.decode(response_ids, skip_special_tokens=False)`.

---

## Optional in-rollout tools (only used in `self_launch_persistent_browser` mode)

Two CLIs installed via `pip install -e .`:

```bash
image_qa --image PATH [--image PATH ...] --question "..."
self_reflection --task "..." --critical-points "1. ..." --action-log <path-or-text> \
    --image PATH [--image PATH ...] --output final_runs/run_<id>/self_reflection.json
```

Both POST to `$WEB_AGENT_POLICY_URL` (the current policy's `/chat/completions`) — so the model self-judges. The URL is exported automatically when training runs with `inference_engine.enable_http_endpoint: true` (already set in `qwen35_4b_web_agent_hard_4gpu_self_launch.yaml`).

---

## Hard-reset before relaunching

If a run hung or died mid-step, clean fully before relaunch:

```bash
ps aux | grep -E "ray|skyrl|vllm" | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 5
rm -rf /tmp/ray /tmp/skyrl_lora_sync /dev/shm/ray /tmp/web_agent_rollouts
nvidia-smi --query-gpu=memory.used --format=csv | head -6   # should show ~4 MiB used
```

---

## Known config quirks (Qwen3.5-4B)

These are baked into `configs/qwen35_4b_web_agent_hard_4gpu.yaml` and the self_launch variant. Don't undo without re-debugging:

- `trainer.use_sample_packing: false` — SkyRL refuses to pack VLM inputs.
- `trainer.flash_attn: false` — flash-attn 2.8.1 hits cudaErrorIllegalAddress on Qwen3.5's varlen attention path; sdpa is the workaround.
- `tilelang` must be installed — `flash-linear-attention` errors on Hopper + Triton 3.4+ for `gated chunk_bwd_dqkwg` and demands tilelang.
- `engine_init_kwargs.model: Qwen/Qwen3.5-4B` (self_launch config only) — needed when `enable_http_endpoint: true` so the OpenAI-style endpoint knows the model id.
- `transformers==5.3.0` — required so the `qwen3_5` model type is recognized; vllm's `<5` pin is technically violated but works at runtime.
- `huggingface_hub>=1.x` — must match transformers 5.3 (`is_offline_mode` import location).
