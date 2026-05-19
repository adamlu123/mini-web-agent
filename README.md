# web-agent RL (echo-rl + mini-web-agent)

RL training for a Playwright-driving web agent, built as a drop-in extension
of [ECHO / echo-rl](https://github.com/microsoft/echo-rl). The agent emits the
same qwen35 tool-call format as echo-rl's terminal agent, but its "terminal"
is a single browser tab seeded from the task's start URL. Reward comes from
the upstream Online-Mind2Web judge (o4-mini via the phyagi gateway) by
default and is fully pluggable.

This branch lives in
[`adamlu123/mini-web-agent` → `rl`](https://github.com/adamlu123/mini-web-agent/tree/rl)
and shares history with `microsoft/echo-rl@main`. Original echo-rl README
content is preserved further down.

## What got built

```
echo_rl/web_agent/
├── prompts.py                       # system prompt + instance template for the `web` CLI
├── dataset.py                       # WebAgentTaskDataset (om2w JSON or parquet)
├── web_environment.py               # WebAgentEnvironment: Playwright tab + stub mode + `web` CLI
├── web_agent_generator.py           # WebAgentGenerator: subclasses TerminalAgentGenerator
├── entrypoint.py                    # SkyRL entrypoint (WebAgentSkyRLConfig/Exp)
├── rewards/
│   ├── base.py                      # BaseRewardFn protocol + build_reward_fn
│   ├── osw_judge.py                 # OSWJudgeReward (wraps upstream WebJudge_Online_Mind2Web_eval)
│   └── deterministic.py             # DeterministicReward placeholder
└── scripts/
    ├── prepare_om2w_data.py         # JSON → parquet for the dataset
    ├── run_smoke_rollout.py         # one-task rollout sans SkyRL (stub or real env)
    └── run_real_rollout.py          # HF model + Browserbase + judge, no SkyRL needed

configs/qwen35_4b_web_agent.yaml     # echo-rl GRPO baseline retargeted at Qwen3.5-4B + web env
scripts/run_web_agent.sh             # entry script (analog of run_echo_terminal_agent.sh)
tests/test_web_agent_e2e.py          # pytest smoke test (stub + deterministic reward)
docs/web_agent_training.md           # full parameter doc and launch procedure
```

Everything from echo-rl's rollout loop is reused unchanged: token masking,
world-model masks, format warnings, length penalty, SkyRL trajectory shape.
The web-agent module only swaps the env, dataset, and reward.

### How the `web` CLI works

The agent issues regular qwen35 tool calls. Each command is interpreted by
`WebAgentEnvironment.exec` as one of three forms:

1. `web <subcommand> [args]` — built-in browser actions:
   `goto / url / title / snapshot / text / click / fill / press / wait / back /
   forward / reload / screenshot / eval / py`
2. `python -c '<code>'` or `python <file>` — runs Python with `page`,
   `context`, `browser`, `task` already injected (the "execute a python script
   to drive the browser" path from the task spec).
3. Anything else — runs in a subprocess inside the rollout workspace.

Each turn ends with a screenshot, and `actions_history()` /
`screenshot_paths()` are what the judge consumes.

### Reward functions

`echo_rl.web_agent.rewards.build_reward_fn` reads a small dict spec from
config and returns a `BaseRewardFn`. Two implementations ship:

- `name: osw_judge` → `OSWJudgeReward` — calls the upstream
  `om2w_judge.methods.webjudge_online_mind2web.WebJudge_Online_Mind2Web_eval`
  through the o4-mini phyagi endpoint, returns 1.0 / 0.0 per the judge label.
- `name: deterministic` → `DeterministicReward` — substring match on
  `final_response`. Useful as a placeholder until per-task verifiers ship,
  and for offline tests.

Adding a new reward source is implementing one `async score(...) ->
RewardResult` method and registering it in `build_reward_fn`. Nothing else
in the rollout loop needs to change.

### Training config

`configs/qwen35_4b_web_agent.yaml` keeps every hyperparameter from
`configs/qwen3_8b_rl.yaml` (the echo-rl GRPO baseline) and only changes
model, dataset, and the reward / env blocks. Full parameter table lives in
[`docs/web_agent_training.md`](docs/web_agent_training.md).

Key knobs:

| Field | Value |
| --- | --- |
| `tokenizer_path` / `policy.model.path` | `Qwen/Qwen3.5-4B` |
| `placement.policy_num_nodes` × `policy_num_gpus_per_node` | 1 × 8 |
| `optimizer_config.lr` | `1.0e-6` |
| `algorithm.max_seq_len` | `34096` |
| `algorithm.world_model_coeff` | `0.0` (vanilla GRPO; set >0 for ECHO) |
| `trainer.train_batch_size` / `policy_mini_batch_size` | 16 / 16 |
| `generator.n_samples_per_prompt` | 16 |
| `generator.max_turns` | 16 |
| `generator.parser_name` | `qwen35` |
| `generator.reward.name` | `osw_judge` (o4-mini via phyagi) |

## Verification

### Stub smoke test (no GPU / network / judge)

```
python -m echo_rl.web_agent.scripts.run_smoke_rollout --stub --deterministic
pytest tests/test_web_agent_e2e.py
```

Both pass. The smoke test exercises dataset loading, the scripted rollout
loop, and the reward function against the in-memory stub browser.

### Real end-to-end test (Qwen3.5-4B + Browserbase + o4-mini judge)

Loads `Qwen/Qwen3.5-4B` with transformers in bf16 on one H100, drives a real
Browserbase Chromium cloud session, and scores the rollout with the upstream
OSW judge through the phyagi `/api/responses` gateway.

```bash
source /home/luyadong/sandbox/cred.sh        # BROWSERBASE_*, HF_TOKEN, OPENAI_GATEWAY_ENDPOINT
source /home/luyadong/cred_gateway.sh        # working OPENAI_API_KEY for the gateway
unset OPENAI_GATEWAY_API_KEY                 # force OpenaiEngine to fall back to OPENAI_API_KEY

python -m echo_rl.web_agent.scripts.run_real_rollout \
    --task-id smoke-001 \
    --task "Open https://example.com and report the page title." \
    --start-url https://example.com \
    --max-turns 4 --max-new-tokens 512
```

Results from real runs against a Browserbase session:

| Task | Stop reason | Judge label | Reward |
| --- | --- | --- | --- |
| Open https://example.com and report the page title | `done` in 2 turns | success (1) | **1.0** |
| Find the FlightAware community discussions link *(initial run)* | `max_turns` (selector quoting bug) | failure (0) | **0.0** |
| Same task, after the `_unquote` env fix | `done` in 5 turns, lands on `https://discussions.flightaware.com/` | success (1) | **1.0** |

The judge correctly distinguished success from failure — it did not reward
the broken-selector run. Sample judge response (FlightAware, success):

> Thoughts: The agent located the "Discussion" link under the COMMUNITY
> section on the FlightAware main page, clicked it, and confirmed navigation
> to the FlightAware community discussions page at
> https://discussions.flightaware.com/ with the correct page title. All key
> points are satisfied.
> Status: success

### Issues hit while verifying, and fixes

| # | Problem | Fix |
| --- | --- | --- |
| 1 | `device_map=` required `accelerate` (not installed). | Loaded with `.to(device)` instead. |
| 2 | `torch_dtype` deprecated in transformers 5.8. | Renamed to `dtype`. |
| 3 | `apply_chat_template(..., return_tensors='pt')` returned a `BatchEncoding`, not a tensor; `.shape` failed. | Added `return_dict=False`. |
| 4 | cuDNN SDPA: *"No valid execution plans built"* on Qwen3.5 attention shape. | `torch.backends.cuda.enable_cudnn_sdp(False)` + `attn_implementation="sdpa"`. |
| 5 | Judge 401: `OPENAI_GATEWAY_API_KEY` in `cred.sh` was rotated and rejected. | Working key lives in `/home/luyadong/cred_gateway.sh` as `OPENAI_API_KEY`; documented setup (source both files, unset the stale var). |
| 6 | Qwen3.5-4B emits shell-quoted selectors like `web click 'text=Discussion'`; my handler passed them verbatim to Playwright. | Added `_unquote()` on `web click / wait / fill` args. Re-ran the FlightAware task — solved correctly the second time. |

## What's intentionally **not** verified yet

Real on-cluster SkyRL training with vLLM + Qwen3.5-4B + the patched SkyRL
hooks was *not* run from this branch. The `run_real_rollout.py` path proves
every component works in isolation (model → browser → judge), and the SkyRL
entrypoint (`echo_rl.web_agent.entrypoint:main`) plus the
`configs/qwen35_4b_web_agent.yaml` config are wired up to slot into the
existing echo-rl training stack — but actually launching a Ray + vLLM training
run is a follow-up.

## Quick start (training)

```
# 1. Build the dataset
python -m echo_rl.web_agent.scripts.prepare_om2w_data \
    --input  /home/luyadong/sandbox/nano_eval/task_files/om2w_260220.json \
    --output ${ECHO_RL_DATA}/web_agent/om2w_train.parquet \
    --val-output ${ECHO_RL_DATA}/web_agent/om2w_val.parquet \
    --train-size 60 --val-size 8

# 2. Set up creds (Browserbase + phyagi judge)
source /home/luyadong/sandbox/cred.sh
source /home/luyadong/cred_gateway.sh
unset OPENAI_GATEWAY_API_KEY

# 3. Launch training
export OUTPUT_DIR=/path/to/outputs/qwen35_4b_web_agent
export CONFIG_PATH=configs/qwen35_4b_web_agent.yaml
./scripts/run_web_agent.sh
```

See [`docs/web_agent_training.md`](docs/web_agent_training.md) for the full
parameter table and the SkyRL setup procedure (patch + install).

## Design

`WebAgentGenerator` subclasses `TerminalAgentGenerator` and overrides only
two seams:

1. `generate(...)` — swaps `HarborEnvironmentProvider` for
   `WebAgentEnvironmentProvider` (per-rollout Playwright tab; no shared image
   build, so `prepare_batch` is a no-op).
2. `_run_one(...)` — replaces the in-container verifier with a call to the
   modular reward function built from `generator.reward` in the YAML.

The rest of the rollout loop — generation, parsing, masking, ECHO-style
auxiliary-loss bookkeeping, format-warning collection, length penalty,
trajectory serialization for SkyRL — is the same code path the terminal
agent uses, so all of echo-rl's training infrastructure carries over
unchanged.

---

# Upstream: ECHO — Terminal Agents Learn World Models for Free

[Paper (PDF)](echo.pdf)

ECHO is an environment cross-entropy hybrid objective, which trains terminal
agents by combining policy-gradient RL with an on-policy cross-entropy loss
for predicting environment tokens.

ECHO is implemented as an extension on top of
[SkyRL](https://github.com/NovaSky-AI/SkyRL): SkyRL provides the core RL
training stack, while this repo adds the terminal-agent integration,
environment prediction loss, example configs, and a small SkyRL hook patch.

![ECHO terminal-agent rollout](echo.gif)

## Quick Start

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git
git clone https://github.com/microsoft/echo-rl.git

cd SkyRL
git checkout 43aab09782953cc7cfc93bda52b1635d717ce446

conda create -n echo-rl python=3.12 -y
conda activate echo-rl
pip install -e ".[fsdp]"
pip install -e ../echo-rl
```

Apply the SkyRL patch:

```bash
cd ../echo-rl
./scripts/apply_hooks_to_skyrl.sh ../SkyRL
```

## Run

Edit the parquet paths in the config you want to run.

Vanilla GRPO:

```bash
cd ../SkyRL
export OUTPUT_DIR=/path/to/outputs/qwen3_8b_rl
export CONFIG_PATH=echo_configs/qwen3_8b_rl.yaml
./run_echo_terminal_agent.sh
```

ECHO (GRPO + Environment Prediction Loss):

```bash
cd ../SkyRL
export OUTPUT_DIR=/path/to/outputs/qwen3_8b_rl_wm05
export CONFIG_PATH=echo_configs/qwen3_8b_rl_wm05.yaml
./run_echo_terminal_agent.sh
```

Checkpoints go to `${OUTPUT_DIR}/ckpts`; logs go to `${OUTPUT_DIR}/skyrl_logs`.

## License

This project is licensed under the [MIT License](LICENSE).
