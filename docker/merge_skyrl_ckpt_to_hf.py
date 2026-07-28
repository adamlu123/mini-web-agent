"""Offline CPU merger: SkyRL FSDP2 (DTensor) sharded policy ckpt -> HF safetensors.

The SkyRL trainer saves SHARDED_STATE_DICT per rank:
    ckpts/global_step_N/policy/model_world_size_W_rank_R.pt
Each value is a DTensor with placements (Shard(dim=0),) over the fsdp mesh and
the GLOBAL shape available on the object; keys match the HF weight_map exactly
(verified against the base last_hf: 760/760 identical). Reconstruction is
therefore: concat rank-ordered local shards on the shard dim, trim padding to
the global shape, cast to the target dtype, and re-shard into safetensors.

Pure CPU, no process group needed (torch 2.10 deserializes DTensor offline and
to_local() works). Peak RAM ~= all rank files + one assembled dict (~75GB for
the 9B VLM) — run it on a node with >=128GB.

Usage (inside any pod that mounts the PVC, e.g. the training master):
    python merge_skyrl_ckpt_to_hf.py \
        --policy-dir /mnt/pvc/.../ckpts/global_step_13/policy \
        --base /mnt/pvc/experiments/luyadong/.../train/last_hf \
        --out /mnt/pvc/.../exports/hf/global_step_13
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import torch

SHARD_MAX_BYTES = 4_800_000_000  # ~4.8GB per safetensors shard


def load_rank_shards(policy_dir: str) -> list[dict]:
    files = sorted(
        (f for f in os.listdir(policy_dir) if re.fullmatch(r"model_world_size_\d+_rank_\d+\.pt", f)),
        key=lambda f: int(re.search(r"rank_(\d+)\.pt", f).group(1)),
    )
    if not files:
        raise SystemExit(f"no model_world_size_*_rank_*.pt under {policy_dir}")
    world = int(re.search(r"world_size_(\d+)_", files[0]).group(1))
    if len(files) != world:
        raise SystemExit(f"expected {world} rank files, found {len(files)} — ckpt incomplete?")
    shards = []
    for i, f in enumerate(files):
        t0 = time.time()
        shards.append(torch.load(os.path.join(policy_dir, f), map_location="cpu", weights_only=False))
        print(f"[load] rank {i}/{world - 1} ({time.time() - t0:.1f}s)", flush=True)
    return shards


def assemble(shards: list[dict], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    keys = list(shards[0].keys())
    full: dict[str, torch.Tensor] = {}
    for n, key in enumerate(keys):
        v0 = shards[0][key]
        if type(v0).__name__ == "DTensor":
            placement = v0.placements[0]
            global_shape = tuple(v0.shape)
            if getattr(placement, "dim", None) is not None and placement.is_shard():
                dim = placement.dim
                locals_ = [s[key].to_local() for s in shards]
                cat = torch.cat(locals_, dim=dim)
                # FSDP2 pads the last chunk for divisibility; trim to global.
                if cat.shape[dim] != global_shape[dim]:
                    cat = cat.narrow(dim, 0, global_shape[dim])
                tensor = cat
            else:  # Replicate
                tensor = v0.to_local()
            if tuple(tensor.shape) != global_shape:
                raise SystemExit(f"shape mismatch for {key}: got {tuple(tensor.shape)}, want {global_shape}")
        else:
            tensor = v0
        full[key] = tensor.to(dtype).contiguous()
        for s in shards:  # free incrementally
            s.pop(key, None)
        if n % 100 == 0:
            print(f"[assemble] {n}/{len(keys)} {key}", flush=True)
    return full


def save_safetensors_sharded(full: dict[str, torch.Tensor], out_dir: str) -> None:
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    groups: list[list[str]] = [[]]
    size = 0
    for key, t in full.items():
        nbytes = t.numel() * t.element_size()
        if size + nbytes > SHARD_MAX_BYTES and groups[-1]:
            groups.append([])
            size = 0
        groups[-1].append(key)
        size += nbytes
    total = len(groups)
    weight_map = {}
    total_size = 0
    for i, group in enumerate(groups, 1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file({k: full[k] for k in group}, os.path.join(out_dir, name))
        for k in group:
            weight_map[k] = name
            total_size += full[k].numel() * full[k].element_size()
        print(f"[save] {name} ({len(group)} tensors)", flush=True)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)


def copy_base_assets(base: str, out_dir: str) -> None:
    for name in os.listdir(base):
        if ".safetensors" in name or name.startswith(("model-", "pytorch_model")):
            continue
        src = os.path.join(base, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            print(f"[copy] {name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--base", required=True, help="HF dir for config/tokenizer + key sanity check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tmp_out = args.out.rstrip("/") + ".tmp"
    if os.path.isdir(tmp_out):
        shutil.rmtree(tmp_out)

    shards = load_rank_shards(args.policy_dir)
    full = assemble(shards, dtype)
    del shards

    # Sanity: key set must equal the base HF weight map.
    base_index = os.path.join(args.base, "model.safetensors.index.json")
    if os.path.isfile(base_index):
        hf_keys = set(json.load(open(base_index))["weight_map"])
        if set(full) != hf_keys:
            missing, extra = hf_keys - set(full), set(full) - hf_keys
            raise SystemExit(f"key mismatch vs base: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
        print(f"[sanity] {len(full)} keys match base weight_map", flush=True)

    save_safetensors_sharded(full, tmp_out)
    copy_base_assets(args.base, tmp_out)
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.rename(tmp_out, args.out)  # atomic-ish publish
    print(f"[done] HF export at {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
