# ECHO: Terminal Agents Learn World Models for Free

[Paper (PDF)](echo.pdf)

ECHO is an environment cross-entropy hybrid objective, which trains terminal agents by combining policy-gradient RL with an on-policy cross-entropy loss for predicting environment tokens.

ECHO is implemented as an extension on top of [SkyRL](https://github.com/NovaSky-AI/SkyRL): SkyRL provides the core RL training stack, while this repo adds the terminal-agent integration, environment prediction loss, example configs, and a small SkyRL hook patch.

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

## Design

The terminal-agent code in `echo_rl/terminal_agent/` defines the task dataset, prompt formatting, tool-call parsing, rollout loop, and SkyRL generator. This code is responsible for turning model generations into terminal commands, tracking the resulting interaction transcript, constructing the token masks used by ECHO, and returning trajectories in the format SkyRL expects for RL training. Harbor is used underneath as the terminal task backend: it starts the task containers, runs commands inside them, returns terminal observations, and executes the verifier that produces task rewards.

We use SkyRL/vLLM for model generation rather than letting Harbor own the full rollout loop because training needs fast, batched, token-level control over generated trajectories. SkyRL needs the generated token ids, logprobs, attention masks, and ECHO-specific environment-token masks to compute both the GRPO objective and the auxiliary environment-prediction loss. Keeping generation in SkyRL/vLLM also gives direct control over batching, sampling, weight synchronization, and trajectory construction.

The world-modeling code in `echo_rl/world_modeling/` adds the ECHO auxiliary loss. During policy training, it computes cross entropy on selected environment-observation tokens and combines that loss with SkyRL's policy-gradient objective. Setting `world_model_coeff: 0.0` recovers the vanilla GRPO baseline; setting it to a positive value enables ECHO.

The patch in `patches/skyrl_minimal_hooks.patch` keeps the SkyRL changes small. It exposes hooks for custom tokenizer paths, custom trainer and worker classes, extra training tensors, zero-padding metadata for auxiliary tensors, token-only inference, and auxiliary policy losses. ECHO uses these hooks to pass world-model masks through SkyRL's batches and add the auxiliary loss in the FSDP policy-gradient path, while leaving the main SkyRL RL implementation in SkyRL.

## License

This project is licensed under the [MIT License](LICENSE).
