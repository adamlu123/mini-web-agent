#!/usr/bin/env python
"""Complete a language-only Qwen3.5 SFT checkpoint by pulling the MISSING tensors
(the vision tower) from the base Qwen3.5-9B, so it loads as the full
Qwen3_5ForConditionalGeneration.

vLLM has no text-only Qwen3.5 causal-LM class, so a text-only SFT ckpt (which
saved only `model.language_model.*` + `lm_head`) fails to load -- the loader
demands the `visual.*` weights. We do text-only inference and never run the
vision tower, but it must be present. The base tower is unchanged by text SFT,
so we copy it verbatim.

This writes a small `vision.safetensors` + a `model.safetensors.index.json` into
the ckpt dir; the 18GB `model.safetensors` (the trained LM) is left untouched, so
only the two small new files need to be shipped to a dev box.

Run on the pod (base is cached in HF_HOME there):
    python scripts/merge_vision_from_base.py \
        --ckpt /mnt/pvc/$USER/models/qwen35_9b/full/websft \
        --base "$(echo /mnt/pvc/$USER/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*)"
"""
import argparse
import glob
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file


def keys_of(path):
    with safe_open(path, framework="pt") as h:
        return list(h.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="language-only SFT ckpt dir (has model.safetensors)")
    ap.add_argument("--base", required=True, help="base model dir/snapshot that has the vision weights")
    args = ap.parse_args()

    lm_files = sorted(
        f for f in glob.glob(os.path.join(args.ckpt, "model*.safetensors"))
        if os.path.basename(f) != "vision.safetensors"
    )
    assert lm_files, f"no model*.safetensors in {args.ckpt}"
    existing = set()
    for f in lm_files:
        existing.update(keys_of(f))
    print(f"[merge] ckpt LM tensors: {len(existing)} across {len(lm_files)} file(s)")

    base_files = sorted(glob.glob(os.path.join(args.base, "*.safetensors")))
    assert base_files, f"no *.safetensors in {args.base}"

    # Copy every base tensor we don't already have (= the vision tower).
    missing = {}
    for f in base_files:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k not in existing:
                    missing[k] = h.get_tensor(k)
    print(f"[merge] tensors to copy from base (missing in ckpt): {len(missing)}")
    assert missing, "base has no tensors missing from ckpt -- nothing to merge (naming mismatch?)"
    prefixes = sorted({k.split('.')[0] + ('.' + k.split('.')[1] if '.' in k else '') for k in missing})
    print(f"[merge] sample missing keys: {list(missing)[:4]}")

    vpath = os.path.join(args.ckpt, "vision.safetensors")
    save_file(missing, vpath, metadata={"format": "pt"})
    print(f"[merge] wrote {vpath} ({os.path.getsize(vpath) / 1e9:.2f} GB)")

    # Build the index over LM file(s) + vision.safetensors so the loader reads both.
    weight_map = {}
    for f in lm_files:
        for k in keys_of(f):
            weight_map[k] = os.path.basename(f)
    for k in keys_of(vpath):
        weight_map[k] = "vision.safetensors"
    total = sum(os.path.getsize(f) for f in lm_files) + os.path.getsize(vpath)
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    ipath = os.path.join(args.ckpt, "model.safetensors.index.json")
    json.dump(index, open(ipath, "w"), indent=2)
    print(f"[merge] wrote {ipath} ({len(weight_map)} keys)")
    print("[merge] ship to dev box: vision.safetensors + model.safetensors.index.json "
          "(model.safetensors is unchanged / already there)")


if __name__ == "__main__":
    main()
