# Web-agent RL training

This doc describes the web-agent RL training added on top of echo-rl. The
implementation lives entirely under `echo_rl/web_agent/` and reuses the
echo-rl terminal-agent rollout loop / SkyRL hooks unchanged.

## What's the same as echo-rl terminal-agent

Everything below is inherited:

- The agent emits qwen35-style `<tool_call>` blocks (one bash command per
  turn) and we feed the terminal output back in as the next observation.
- The token-level rollout loop, masking, world-modeling masks, length penalty,
  format warnings, and SkyRL trajectory shape come from
  `echo_rl.terminal_agent.TerminalAgentGenerator`.
- All optimizer / SkyRL knobs in the YAML config are identical to the
  echo-rl GRPO baseline (`configs/qwen3_8b_rl.yaml`). The only differences
  are the model path, the dataset, and the reward.

## What's different

| Concern        | Terminal-agent                                  | Web-agent                                                               |
| -------------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| Env            | Harbor docker container (per task image)        | Single Playwright tab (per rollout), seeded from `start_url`            |
| Command        | Any shell command in the container              | `web <subcommand>` browser actions; `python -c '...'`; plain shell      |
| Verifier       | In-container task verifier                      | Modular `BaseRewardFn` — defaults to `OSWJudgeReward` (o4-mini phyagi)  |
| Data shape     | `task_binary` tar of a Docker task              | om2w-style JSON / parquet (task id, prompt, start URL, level)           |
| Entrypoint     | `echo_rl.terminal_agent.entrypoint:main`        | `echo_rl.web_agent.entrypoint:main`                                     |

## Quick start

### 1. Generate the parquets

```
python -m echo_rl.web_agent.scripts.prepare_om2w_data \
    --input  /home/luyadong/sandbox/nano_eval/task_files/om2w_260220.json \
    --output ${ECHO_RL_DATA}/web_agent/om2w_train.parquet \
    --val-output ${ECHO_RL_DATA}/web_agent/om2w_val.parquet \
    --train-size 60 --val-size 8
```

The parquet only stores task metadata (`task_id`, `task`, `start_url`,
`level`, `reference_length`). `WebAgentTaskDataset` tokenizes the prompt at
load time using the chat template configured in the YAML.

### 2. Smoke-test the rollout loop

Stub mode (no internet, no GPU, no judge — runs in seconds):

```
python -m echo_rl.web_agent.scripts.run_smoke_rollout --stub --deterministic
```

`pytest tests/test_web_agent_e2e.py` runs the same checks under pytest using
the in-memory stub browser, no network required.

#### Real end-to-end run (Browserbase + Qwen3.5-4B + o4-mini judge)

This loads the policy from HF, drives a real Browserbase Chromium session,
and scores the rollout with the upstream OSW judge. Verified working — see
`/tmp/web_agent_real_rollout/transcript_*.json` for sample transcripts.

```
source /home/luyadong/sandbox/cred.sh        # BROWSERBASE_*, HF_TOKEN, OPENAI_GATEWAY_ENDPOINT
source /home/luyadong/cred_gateway.sh        # working OPENAI_API_KEY for the gateway
unset OPENAI_GATEWAY_API_KEY                 # force OpenaiEngine to fall back to OPENAI_API_KEY

python -m echo_rl.web_agent.scripts.run_real_rollout \
    --task-id smoke-001 \
    --task "Open https://example.com and report the page title." \
    --start-url https://example.com \
    --max-turns 4 --max-new-tokens 512
```

Observed (Qwen3.5-4B, bf16, single H100):
- `example.com` title-report task → reward 1.0, judge `predicted_label=1`.
- FlightAware "find the discussions link" task → reward 1.0 after Qwen3.5-4B
  navigates, clicks the right link, and the judge confirms the destination
  `https://discussions.flightaware.com/`.

If you only set `cred.sh`, `OpenaiEngine` will pick up the stale
`OPENAI_GATEWAY_API_KEY` and 401 on the judge call. Either source
`cred_gateway.sh` after `cred.sh` and unset the stale var (as above), or pass
`api_key=...` directly inside the `reward:` block of the YAML.

### 3. Launch SkyRL training

```
export OUTPUT_DIR=/path/to/outputs/qwen35_4b_web_agent
export CONFIG_PATH=configs/qwen35_4b_web_agent.yaml
export OPENAI_GATEWAY_API_KEY=...
export OPENAI_GATEWAY_ENDPOINT=http://gateway.phyagi.net/api/responses
# Playwright must be installed inside the training container/conda env:
#   pip install playwright && playwright install chromium

./scripts/run_web_agent.sh
```

Checkpoints land in `${OUTPUT_DIR}/ckpts`, logs in `${OUTPUT_DIR}/skyrl_logs`.

## Run parameters (defaults inherited from echo-rl GRPO baseline)

These come from `configs/qwen3_8b_rl.yaml` unchanged. The only swaps are the
model path, dataset paths, reward block, and `env_overrides`.

### Model & tokenizer

| Field                    | Value             | Where    |
| ------------------------ | ----------------- | -------- |
| `tokenizer_path`         | `Qwen/Qwen3.5-4B` | trainer  |
| `trainer.policy.model.path` | `Qwen/Qwen3.5-4B` | trainer |
| `chat_template_path`     | `echo_rl/terminal_agent/chat_templates/qwen3_xml_tool_calling.jinja` | trainer |

### Placement / parallelism

| Field                              | Value |
| ---------------------------------- | ----- |
| `placement.policy_num_nodes`       | 1     |
| `placement.policy_num_gpus_per_node` | 8 |
| `placement.colocate_all`           | true  |
| `placement.colocate_policy_ref`    | true  |

### Optimizer

| Field                                  | Value                  |
| -------------------------------------- | ---------------------- |
| `optimizer_config.lr`                  | `1.0e-6`               |
| `optimizer_config.adam_betas`          | `[0.9, 0.999]`         |
| `optimizer_config.weight_decay`        | `0.01`                 |
| `optimizer_config.max_grad_norm`       | `0.2`                  |
| `optimizer_config.scheduler`           | `constant_with_warmup` |
| `optimizer_config.num_warmup_steps`    | `20`                   |

### Algorithm (GRPO baseline)

| Field                                 | Value                       |
| ------------------------------------- | --------------------------- |
| `algorithm.max_seq_len`               | `34096`                     |
| `algorithm.use_kl_loss`               | `false`                     |
| `algorithm.use_kl_in_reward`          | `false`                     |
| `algorithm.kl_loss_coef`              | `0.0`                       |
| `algorithm.eps_clip_low` / `eps_clip_high` | `0.2` / `0.2`          |
| `algorithm.advantage_batch_normalize` | `false`                     |
| `algorithm.loss_reduction`            | `sequence_mean`             |
| `algorithm.world_model_coeff`         | `0.0` (vanilla GRPO; set >0 for ECHO) |
| `algorithm.world_loss_normalization`  | `full_observation_tokens`   |

### Training loop

| Field                              | Value |
| ---------------------------------- | ----- |
| `trainer.epochs`                   | `2`   |
| `trainer.update_epochs_per_batch`  | `1`   |
| `trainer.max_prompt_length`        | `1536`|
| `trainer.train_batch_size`         | `16`  |
| `trainer.policy_mini_batch_size`   | `16`  |
| `trainer.micro_train_batch_size_per_gpu` | `1` |
| `trainer.micro_forward_batch_size_per_gpu` | `1` |
| `trainer.eval_batch_size`          | `16`  |
| `trainer.eval_interval`            | `20`  |
| `trainer.ckpt_interval`            | `20`  |
| `trainer.hf_save_interval`         | `-1`  |
| `trainer.logger`                   | `wandb` |
| `trainer.project_name`             | `world_model` |
| `trainer.run_name`                 | `web_agent_qwen35_4b` |

### Generator / rollout

| Field                              | Value         |
| ---------------------------------- | ------------- |
| `n_samples_per_prompt`             | `16`          |
| `eval_n_samples_per_prompt`        | `8`           |
| `parser_name`                      | `qwen35`      |
| `command_selection`                | `first` (one command per turn) |
| `max_turns`                        | `16`          |
| `max_context_tokens`               | `16384`       |
| `max_tokens_per_generation`        | `2048`        |
| `max_total_tokens`                 | `34096`       |
| `max_terminal_output_chars`        | `50000`       |
| `terminal_output_truncation`       | `start`       |
| `add_format_warn`                  | `true`        |
| `world_loss_target`                | `env_only`    |
| `thinking_handling`                | `keep_all`    |
| `verifier_timeout`                 | `120.0`       |
| `length_penalty_coef`              | `0.0`         |
| `length_penalty_threshold`         | `20000`       |
| `correct_threshold`                | `0.5`         |
| `agent_max_concurrency`            | `32`          |
| `agent_timeout` (sec)              | `1200.0`      |
| `sampling_params.temperature`      | `0.8`         |
| `eval_sampling_params.temperature` | `0.6`         |

### Inference engine

| Field                                | Value   |
| ------------------------------------ | ------- |
| `inference_engine.backend`           | `vllm`  |
| `inference_engine.num_engines`       | `8`     |
| `inference_engine.tensor_parallel_size` | `1` |
| `inference_engine.async_engine`      | `true`  |
| `inference_engine.enable_prefix_caching` | `true` |
| `inference_engine.enable_chunked_prefill` | `true` |
| `inference_engine.gpu_memory_utilization` | `0.8` |
| `inference_engine.max_num_batched_tokens` | `262144` |
| `inference_engine.engine_init_kwargs.swap_space` | `4` |

### Web-agent specific knobs

| Field                                 | Value                                  |
| ------------------------------------- | -------------------------------------- |
| `generator.reward.name`               | `osw_judge`                            |
| `generator.reward.judge_model`        | `o4-mini`                              |
| `generator.reward.judge_gateway_endpoint` | `http://gateway.phyagi.net/api/responses` |
| `generator.reward.score_threshold`    | `3`                                    |
| `generator.reward.mini_web_agent_root`| `/home/luyadong/sandbox/mini-web-agent`|
| `generator.env_overrides.headless`    | `true`                                 |
| `generator.env_overrides.nav_timeout_ms` | `30000`                             |
| `generator.env_overrides.op_timeout_ms`  | `15000`                             |
| `generator.env_overrides.max_snapshot_chars` | `6000`                          |
| `generator.env_overrides.workspace_root` | `/tmp/web_agent_rollouts`           |
| `generator.stub_env`                  | `false`                                |

## Swapping the reward function

The reward is built from the `generator.reward` block via
`echo_rl.web_agent.rewards.build_reward_fn`. Out of the box:

- `name: osw_judge` → `OSWJudgeReward`: calls the upstream
  `om2w_judge.methods.webjudge_online_mind2web.WebJudge_Online_Mind2Web_eval`
  through the o4-mini phyagi endpoint. Returns 1.0 for success, 0.0
  otherwise.
- `name: deterministic` → `DeterministicReward`: substring match against the
  agent's `final_response`. Intended as a placeholder until per-task
  verifiers ship; useful for offline tests too.

To add a new reward source, implement `BaseRewardFn.score` (returns a
`RewardResult`) and register it inside `build_reward_fn`. Nothing else in the
rollout loop has to change.

## Files added by this branch

```
echo_rl/web_agent/
├── __init__.py
├── prompts.py                       # system prompt + instance template for `web` CLI
├── dataset.py                       # WebAgentTaskDataset (om2w JSON or parquet)
├── web_environment.py               # WebAgentEnvironment (Playwright + stub mode)
├── web_agent_generator.py           # WebAgentGenerator (subclasses TerminalAgentGenerator)
├── entrypoint.py                    # SkyRL entrypoint (WebAgentSkyRLConfig/Exp)
├── rewards/
│   ├── __init__.py
│   ├── base.py                      # BaseRewardFn protocol + build_reward_fn
│   ├── osw_judge.py                 # OSWJudgeReward (wraps upstream WebJudge)
│   └── deterministic.py             # DeterministicReward placeholder
└── scripts/
    ├── prepare_om2w_data.py         # JSON → parquet for the dataset
    ├── run_smoke_rollout.py         # one-task rollout sans SkyRL (stub or real env)
    └── run_real_rollout.py          # real rollout: HF model + Browserbase + judge

configs/qwen35_4b_web_agent.yaml     # GRPO baseline retargeted at Qwen3.5-4B + web env
scripts/run_web_agent.sh             # entry script (analog of run_echo_terminal_agent.sh)
tests/test_web_agent_e2e.py          # pytest-based smoke test (stub + deterministic)
```
