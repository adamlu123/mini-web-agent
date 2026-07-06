"""Minimal full-finetune SFT loop for Qwen3.5-4B on web-agent trajectories.

Single-GPU, plain PyTorch (no trl/peft/accelerate launcher needed). Trains only on
assistant turns; see ``data.py`` for the masking. Defaults are sized for one H100.

Example:
    /home/luyadong/.venv/bin/python -m miniswewebagent.trainer.train \
        --data-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0531/pae_100 \
        --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/sft/qwen35_4b_pae100
"""

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import Collator, load_examples


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data-dir", default="/home/luyadong/sandbox/mini-web-agent/outputs/default/0531/pae_100")
    p.add_argument("--output-dir", default="/home/luyadong/sandbox/mini-web-agent/outputs/sft/qwen35_4b_pae100")
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="cap number of training examples (0 = all)")
    p.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0 = no cap)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"

    print(f"[load] tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)
    model.train()

    print(f"[data] loading trajectories from {args.data_dir}")
    examples = load_examples(args.data_dir, tokenizer, max_len=args.max_len)
    if args.limit:
        examples = examples[: args.limit]
    sup = sum(sum(1 for l in ex["labels"] if l != -100) for ex in examples)
    tot = sum(len(ex["input_ids"]) for ex in examples)
    print(f"[data] {len(examples)} trajectories | {tot} tokens | {sup} supervised ({100*sup/max(tot,1):.1f}%)")

    collate = Collator(tokenizer.pad_token_id)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps
    )
    print(f"[train] {args.epochs} epochs | {total_steps} optimizer steps | lr {args.lr}")

    step = 0
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    print(
                        f"[train] epoch {epoch} step {step}/{total_steps} "
                        f"loss {out.loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                if args.max_steps and step >= args.max_steps:
                    print("[train] reached --max-steps; stopping early")
                    break
        if args.max_steps and step >= args.max_steps:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[save] {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[done]")


if __name__ == "__main__":
    main()
