#!/usr/bin/env python3
"""Package ShareGPT JSON + referenced image files for cluster training.

Copies all images referenced by a dataset JSON into an assets directory and
rewrites `images` entries to relative paths. The resulting JSON can be trained
with LlamaFactory by setting `dataset_dir` and `media_dir` to the bundle root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def image_key(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    suffix = path.suffix.lower() or ".png"
    return f"{digest}_{path.name if path.name.endswith(suffix) else path.stem + suffix}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("LlamaFactory/data/web_agent_unified_mix.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("LlamaFactory/data/web_agent_unified_mix_bundle"))
    parser.add_argument("--json-name", default="web_agent_unified_mix_portable.json")
    parser.add_argument("--dataset-name", default="web_agent_unified_mix_portable")
    parser.add_argument("--images-subdir", default="images")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8", errors="replace"))
    out_dir = args.out_dir
    image_dir = out_dir / args.images_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    missing: list[str] = []
    total_refs = 0

    for sample in data:
        images = sample.get("images") or []
        if not images:
            continue
        new_images: list[str] = []
        for raw in images:
            total_refs += 1
            src = Path(str(raw)).expanduser()
            if not src.is_file():
                missing.append(str(raw))
                new_images.append(str(raw))
                continue
            key = str(src.resolve())
            if key not in copied:
                dst_name = image_key(src)
                dst = image_dir / dst_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied[key] = f"{args.images_subdir}/{dst_name}"
            new_images.append(copied[key])
        sample["images"] = new_images

    out_json = out_dir / args.json_name
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_info = {
        args.dataset_name: {
            "file_name": args.json_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "source_json": str(args.input.resolve()),
        "output_json": str(out_json.resolve()),
        "dataset_name": args.dataset_name,
        "bundle_root": str(out_dir.resolve()),
        "images_dir": str(image_dir.resolve()),
        "examples": len(data),
        "image_references": total_refs,
        "unique_images_copied": len(copied),
        "missing_images": len(missing),
        "missing_sample": missing[:50],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
