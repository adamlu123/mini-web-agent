#!/usr/bin/env python3
"""Build a mini-web-agent tasks file from ScienceAgentBench (verified split).

Each SAB instance becomes one task record consumable by
`miniswewebagent.run.benchmarks.om2w` (via `normalize_om2w_task`, which reads
`task_id` / `confirmed_task` / `website` / `level`). The `confirmed_task` text is
assembled to match the D3-Gym science task format the D3 SFT data was built from
(make_d3gym_science_sft.py): the task instruction, a `benchmark/datasets/`-rooted
directory tree in `├──/└──` style, and the dataset previews. All SAB-specific
fields needed by the downstream SAB evaluation harness (gold_program_name,
output_fname, eval_script_name, …) are preserved under `sab`.

Paths inside the task text use `benchmark/datasets/<repo>/...`, matching SAB gold
programs. The eval yaml seed-symlinks `benchmark/datasets` into each task
workspace so those paths resolve.

Usage:
  python scripts/make_sab_tasks_json.py \
    --out src/miniswewebagent/run/benchmarks/sab_verified.json
  # add --use-knowledge to append SAB expert domain knowledge to each task
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

# Reuse the EXACT tree renderer the D3 science SFT data was built with, so the
# eval prompt is byte-identical in convention to the training prompt.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LlamaFactory" / "scripts"))
from make_d3gym_science_sft import render_tree  # noqa: E402

DATA_INTRO = (
    "You can access the dataset at `benchmark/`. "
    "Here is the directory structure of the dataset:"
)


def sab_tree_to_rel_paths(sab_tree: str) -> list[str]:
    """Parse SAB's `|--`/`|----` tree into repo-relative file paths (leaves only),
    e.g. `clintox/clintox_train.csv`. Depth = (#dashes)/2, repo dir == depth 1."""
    nodes = []  # (depth, name, ends_with_slash)
    for raw in sab_tree.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip("|")
        n_dash = len(stripped) - len(stripped.lstrip("-"))
        name = stripped[n_dash:].strip()
        nodes.append((n_dash // 2, name.rstrip("/"), name.endswith("/")))
    # full paths via a depth stack
    paths, stack = [], []
    for depth, name, slash in nodes:
        stack = stack[: depth - 1] + [name]
        paths.append(("/".join(stack), depth, slash))
    files = []
    for i, (full, depth, slash) in enumerate(paths):
        has_child = i + 1 < len(paths) and paths[i + 1][1] > depth
        if has_child or slash:
            continue  # directory
        files.append(full)
    return files


def build_task_text(ex: dict, use_knowledge: bool) -> str:
    """Construct the task text EXACTLY like the D3 science SFT data
    (make_d3gym_science_sft.build_task_text): statement + a benchmark/datasets/-
    rooted, alphabetically-sorted box-drawing tree (via the shared render_tree) +
    the [START Preview ...] blocks. Guarantees eval prompt == training prompt."""
    parts = [ex["task_inst"].strip()]
    if use_knowledge and str(ex.get("domain_knowledge", "")).strip():
        parts.append(str(ex["domain_knowledge"]).strip())
    rel_files = sab_tree_to_rel_paths(ex["dataset_folder_tree"])
    if rel_files:
        parts.append(DATA_INTRO + "\n" + render_tree(rel_files))
    preview = str(ex.get("dataset_preview", "")).strip()
    if preview:
        parts.append("Here are some helpful previews for the dataset file(s):\n" + preview)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="verified")
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent
            / "src/miniswewebagent/run/benchmarks/sab_verified.json"
        ),
    )
    ap.add_argument("--use-knowledge", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("osunlp/ScienceAgentBench", split=args.split)
    tasks = []
    for ex in ds:
        iid = str(ex["instance_id"])
        # first tree line, e.g. "|-- clintox/" -> repo dir "clintox"
        repo = ex["dataset_folder_tree"].splitlines()[0].lstrip("|").lstrip("-").strip().rstrip("/")
        tasks.append(
            {
                "task_id": f"sab_{iid}",
                "confirmed_task": build_task_text(ex, args.use_knowledge),
                "website": "",
                "level": "sab",
                "reference_length": 0,
                "sab": {
                    "instance_id": iid,
                    "domain": ex.get("domain", ""),
                    "github_name": ex.get("github_name", ""),
                    "dataset_repo": repo,
                    "gold_program_name": ex["gold_program_name"],
                    "output_fname": ex["output_fname"],
                    "eval_script_name": ex["eval_script_name"],
                    "use_knowledge": args.use_knowledge,
                },
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1))
    print(f"{len(tasks)} 个 SAB 任务 (use_knowledge={args.use_knowledge}) -> {out}")


if __name__ == "__main__":
    main()
