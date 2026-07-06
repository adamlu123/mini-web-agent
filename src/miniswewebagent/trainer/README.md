# Minimal SFT trainer

A dead-simple, single-GPU full-finetune loop for **Qwen3.5-4B** on the web-agent
trajectories under `outputs/default/0531/pae_100`. Plain PyTorch + `transformers`
only — no `trl`, `peft`, or accelerate launcher.

## Files

- `data.py` — loads every `*/trajectory.json`, emits **one example per assistant
  turn** (prompt = history rendered with the chat template, target = that turn's
  strict-JSON action from `extra.raw_response`). Loss is masked to the completion
  only. Long histories are left-truncated to `--max-len` so the target is kept.
- `train.py` — bf16 full finetune, gradient checkpointing, AdamW + cosine schedule.

## Run

```bash
cd /home/luyadong/sandbox/mini-web-agent/src
CUDA_VISIBLE_DEVICES=0 /home/luyadong/.venv/bin/python -m miniswewebagent.trainer.train \
  --data-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0531/pae_100 \
  --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/sft/qwen35_4b_pae100 \
  --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-5 --max-len 16384
```

The data yields **1011 examples** (assistant turns) from 83 trajectories.

### Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 /home/luyadong/.venv/bin/python -m miniswewebagent.trainer.train \
  --max-len 4096 --limit 8 --grad-accum 2 --max-steps 3 --epochs 1 --output-dir /tmp/sft_smoke
```

## Notes / knobs

- `--limit N` caps the number of examples; `--max-steps N` stops after N optimizer
  steps. Both default to 0 (use everything).
- Qwen3.5-4B is a hybrid linear-attention model loaded via
  `AutoModelForImageTextToText`; we train **text-only** (the trajectories carry no
  images). `attn_implementation="eager"` avoids the optional fast-path deps.
- One H100 (80 GB) handles `--max-len 16384`, `--batch-size 1` with gradient
  checkpointing. Raise `--grad-accum` rather than `--batch-size` to grow the
  effective batch without more memory. For long histories most examples hit
  `--max-len`; lower it (e.g. 8192) to train faster on less context.
- The checkpoint is saved with `save_pretrained` (full weights + tokenizer), so it
  can be pointed at directly as a `model_name` for inference.
```
