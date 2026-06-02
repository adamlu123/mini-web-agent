# Web-Agent SFT (Qwen3.5-4B on pae_100 trajectories)

Full-parameter SFT of Qwen3.5-4B on the mini-web-agent online rollouts, converted
to multi-turn ShareGPT. Each assistant turn is `<think>...</think>` followed by a
`<bash>...</bash>` action (or `<answer>...</answer>` to finish the task).

## Files

| File | Role |
|------|------|
| `scripts/make_web_agent_sft.py` | trajectory.json → ShareGPT converter |
| `data/web_agent_pae100.json` | generated dataset (86 convos / ~1005 assistant turns) |
| `data/dataset_info.json` → `web_agent_pae100` | dataset registration (sharegpt) |
| `examples/train_full/qwen35_4b_websft.yaml` | full-train config |
| `examples/train_full/qwen35_4b_websft_smoke.yaml` | 5-step smoke test |

## 1. Build the dataset

```bash
cd LlamaFactory
python scripts/make_web_agent_sft.py \
    --src /data/t-yifeili/sft_data/pae_100 \
    --out data/web_agent_pae100.json
# knobs:
#   --max-obs-chars N     middle-truncate each observation/user turn (default 3000)
#   --no-normalize-paths  keep raw abs paths (NOT recommended)
```

### What the converter does
- **system**: the original strict-JSON harness prompt is dropped and replaced
  with one describing the `<think>`/`<bash>`/`<answer>` format actually trained.
- **human turns**: every `role:user` message (task + compacted history, and each
  Observation) kept verbatim, then middle-truncated to `--max-obs-chars`.
- **gpt turns**: `content` (the model's thought) → `<think>...</think>`; the
  action `extra.actions[0].bash_command` → `<bash>...</bash>`; the final `done`
  turn → `<answer>{final_response}</answer>`.
- **Anti-overfitting normalization** (default on): each task's absolute workspace
  path `/home/<user>/sandbox/mini-web-agent/outputs/.../<Task>` → `/workspace`,
  repo root → `/opt/mini-web-agent`, `/home/<user>` → `/home/agent`, and bare
  username tokens (from `ls -la` owner columns) → `user`. This removes the
  one-machine directory layout **and** the task-name leak embedded in the path.

## 2. Smoke test (verify the data + model path, 5 steps)

```bash
cd LlamaFactory
FORCE_TORCHRUN=1 NPROC_PER_NODE=4 DISABLE_VERSION_CHECK=1 \
  llamafactory-cli train examples/train_full/qwen35_4b_websft_smoke.yaml \
  dataset=web_agent_pae100 template=qwen3_5 cutoff_len=32768
```

## 3. Full training

```bash
cd LlamaFactory
PYTORCH_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 NPROC_PER_NODE=4 DISABLE_VERSION_CHECK=1 \
  llamafactory-cli train examples/train_full/qwen35_4b_websft.yaml
```

## Memory notes (important)

Qwen3.5-4B has a **248k vocab**; the cross-entropy logits tensor is
`seq_len × 248320`, which at `cutoff_len=32768` is the ~22 GiB single allocation
that OOMs plain ZeRO-3 on 80 GB GPUs. Mitigations baked into the config:

- `deepspeed: ds_z3_offload_config.json` — offloads fp32 master weights + Adam
  states (~13 GB/GPU) and params to CPU, freeing the headroom the logits need.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` — reduces fragmentation OOMs.

If it still OOMs, in order of preference:
1. lower `cutoff_len` (32768 → 24576 → 16384); logits memory scales linearly,
   and with `--max-obs-chars 3000` ~92% of convos already fit in 24576.
2. `pip install liger-kernel` and add `enable_liger_kernel: true` (fused CE never
   materializes the full logits) — best fix if the Qwen3.5 family is supported.
3. switch to LoRA (`finetuning_type: lora`).

## Training behavior

- Template `qwen3_5` (`ReasoningTemplate`). With `mask_history: false` the
  per-turn `<think>` blocks are **preserved on every turn** and loss is computed
  on all assistant turns (think + action), human turns masked. `enable_thinking:
  true` ensures no chain-of-thought is stripped.
- Small dataset (86 convos): watch for overfitting — consider 1–2 epochs and a
  held-out eval split rather than the default 3.
