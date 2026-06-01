"""Merge a SkyRL FSDP2 sharded policy checkpoint into HuggingFace safetensors.

SkyRL saves the policy with ``StateDictType.SHARDED_STATE_DICT`` -- one
``model_world_size_{W}_rank_{r}.pt`` per rank, where every value is an FSDP2
``DTensor`` sharded ``Shard(dim=0)`` across a 1-D ``fsdp={W}`` mesh. The eval
entrypoint (``echo_rl.web_agent.eval_entrypoint``) instead loads plain HF
weights into vLLM via ``trainer.policy.model.path``, so to evaluate a trained
checkpoint we first reconstruct the full weights here.

Reconstruction is purely local (no process group / GPUs): for each parameter we
concatenate the per-rank local shards along dim 0 and narrow back to the
DTensor's declared full shape (trimming FSDP's trailing-rank padding). The
merged state dict keys are already canonical HF names, so we write them straight
to sharded safetensors + an index (the same layout ``save_pretrained`` produces)
and copy the checkpoint's own ``config.json`` / tokenizer / chat template
alongside. We deliberately do NOT instantiate the model class: ``qwen3_5`` is a
multimodal config whose ``vocab_size`` lives under ``text_config``, so
``AutoModelForCausalLM.from_config`` rejects it -- and vLLM loads from raw
safetensors + ``config.json`` anyway (that is how the base eval loads weights).

Usage:
    python -m echo_rl.web_agent.scripts.convert_fsdp_ckpt_to_hf \
        --ckpt   /path/to/outputs/<run>/ckpts/global_step_9/policy \
        --output /path/to/outputs/<run>/hf/global_step_9

Then point eval at it:
    MODEL=4b CKPT=/path/to/.../hf/global_step_9 bash run_local_eval.sh
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from huggingface_hub import split_torch_state_dict_into_shards


def _full_from_shards(values):
    """Rebuild one parameter's full tensor from its per-rank DTensor shards."""
    first = values[0]
    # Plain (non-distributed) tensor -- identical on every rank, take as-is.
    if not hasattr(first, "placements"):
        return first.detach().clone()

    placement = first.placements[0]
    locals_ = [v.to_local() if hasattr(v, "to_local") else v._local_tensor for v in values]

    if placement.is_replicate():
        return locals_[0].detach().clone()

    # Sharded -- concat the local chunks along the shard dim, then trim padding
    # back to the DTensor's logical full shape (FSDP2 pads the trailing rank).
    dim = placement.dim
    full = torch.cat([t.detach() for t in locals_], dim=dim)
    return full.narrow(dim, 0, first.shape[dim]).contiguous()


def merge_sharded_state_dict(ckpt_dir, world_size):
    rank_sds = []
    for r in range(world_size):
        path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{r}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing shard: {path}")
        print(f"[convert] loading {os.path.basename(path)}")
        rank_sds.append(torch.load(path, map_location="cpu", weights_only=False))

    keys = list(rank_sds[0].keys())
    merged = {}
    for i, k in enumerate(keys):
        merged[k] = _full_from_shards([sd[k] for sd in rank_sds])
        if (i + 1) % 100 == 0 or i + 1 == len(keys):
            print(f"[convert] merged {i + 1}/{len(keys)} tensors")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the .../policy checkpoint dir")
    ap.add_argument("--output", required=True, help="dir to write the merged HF model to")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="dtype to save weights in (default bfloat16; checkpoint is fp32)")
    ap.add_argument("--aux-from", default=None,
                    help="HF model id or local dir to copy processor/aux files from "
                         "(preprocessor_config.json, video_preprocessor_config.json, "
                         "vocab.json, merges.txt). qwen3_5 is multimodal so vLLM needs "
                         "preprocessor_config.json, which the FSDP ckpt does not store.")
    args = ap.parse_args()

    fsdp_cfg_path = os.path.join(args.ckpt, "fsdp_config.json")
    world_size = json.load(open(fsdp_cfg_path))["world_size"] if os.path.exists(fsdp_cfg_path) else 4
    hf_src = os.path.join(args.ckpt, "huggingface")
    print(f"[convert] ckpt={args.ckpt} world_size={world_size} -> {args.output}")

    merged = merge_sharded_state_dict(args.ckpt, world_size)
    save_dtype = getattr(torch, args.dtype)
    merged = {k: (v.to(save_dtype) if v.is_floating_point() else v).contiguous()
              for k, v in merged.items()}

    os.makedirs(args.output, exist_ok=True)

    # Shard into safetensors files + index exactly like save_pretrained does.
    split = split_torch_state_dict_into_shards(merged, filename_pattern="model{suffix}.safetensors")
    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {name: merged[name] for name in tensor_names}
        print(f"[convert] writing {filename} ({len(shard)} tensors)")
        save_file(shard, os.path.join(args.output, filename), metadata={"format": "pt"})
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print(f"[convert] wrote index for {len(split.filename_to_tensors)} shards")

    # Carry over config + tokenizer + chat template that live alongside the shards.
    for fn in ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "chat_template.jinja", "vocab.json",
               "merges.txt", "special_tokens_map.json", "preprocessor_config.json"):
        src = os.path.join(hf_src, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, fn))

    # qwen3_5 is multimodal -> vLLM needs the processor configs, which the FSDP
    # checkpoint does not store. Pull them from a base model snapshot.
    if args.aux_from:
        from huggingface_hub import hf_hub_download
        aux = ("preprocessor_config.json", "video_preprocessor_config.json",
               "vocab.json", "merges.txt")
        for fn in aux:
            dst = os.path.join(args.output, fn)
            if os.path.exists(dst):
                continue
            try:
                if os.path.isdir(args.aux_from):
                    src = os.path.join(args.aux_from, fn)
                    if os.path.exists(src):
                        shutil.copy2(os.path.realpath(src), dst)
                        print(f"[convert] aux: copied {fn}")
                else:
                    p = hf_hub_download(args.aux_from, fn)
                    shutil.copy2(p, dst)
                    print(f"[convert] aux: fetched {fn} from {args.aux_from}")
            except Exception as e:  # noqa: BLE001 - aux files are best-effort
                print(f"[convert] aux: skip {fn} ({type(e).__name__})")
    print(f"[convert] done -> {args.output}")


if __name__ == "__main__":
    main()
