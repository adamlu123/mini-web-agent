#!/usr/bin/env python3
"""Prepare the RST task pool: download, dedup, band by difficulty, extract.

Reproduces the working set from scratch on a fresh machine:

  1. download `Zhongzhi1228/Recursive-Task-Synthesis` (~3.9 GB)
  2. collapse near-duplicate rewrite variants to one representative per cluster
     (37,484 tasks -> 12,010 clusters; the paper keeps variants undeduplicated
     and says so, which is why this step is here)
  3. take the shortest N by reference-solution length, the paper's own difficulty
     proxy, and split them into equal groups ordered easy -> hard
  4. extract the task packages and write one manifest per group

Multi-container tasks (those shipping a compose file) are excluded: the
`terminal_docker` environment cannot run them and would raise at prepare().

    python prepare_rst_tasks.py --dest /home/luyadong/data/rst

Needs: huggingface_hub, pyarrow, pandas.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import tarfile
from pathlib import Path

COMPOSE_NAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml")
REPO_ID = "Zhongzhi1228/Recursive-Task-Synthesis"


def nonempty_lines(text: str) -> int:
    """The paper's difficulty metric: non-empty lines of solution/solve.sh."""
    return sum(1 for line in (text or "").splitlines() if line.strip())


def download(dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[1/4] downloading {REPO_ID} -> {dest}")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(dest))
    return dest


def select(dest: Path, total: int, groups: int) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    import pyarrow.parquet as pq

    print("[2/4] deduplicating variant clusters and banding by difficulty")
    df = pq.read_table(
        dest / "metadata/tasks.parquet",
        columns=["task_id", "task_group_id", "solution", "task_toml", "member_prefix", "shard"],
    ).to_pandas()
    df["nel"] = df["solution"].map(nonempty_lines)
    df["category"] = df["task_toml"].map(
        lambda t: (re.search(r'^category\s*=\s*"([^"]*)"', t or "", re.M) or [None, None])[1]
    )

    # One representative per cluster: the median-length member, so the choice does
    # not systematically bias the band downward. Ties break on task_id, so the
    # selection is reproducible.
    df = df.sort_values(["task_group_id", "nel", "task_id"])
    df["idx"] = df.groupby("task_group_id").cumcount()
    middle = (df.groupby("task_group_id")["nel"].transform("size") - 1) // 2
    reps = df[df["idx"] == middle].copy()
    print(f"      {len(df):,} tasks -> {len(reps):,} clusters")

    reps = reps.sort_values(["nel", "task_id"]).head(total).reset_index(drop=True)
    per_group = len(reps) // groups
    reps["group"] = pd.Series(reps.index // per_group + 1).clip(upper=groups).to_numpy()
    return reps


def extract(dest: Path, reps, out_root: Path, groups: int) -> None:
    tasks_dir = out_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    print("[3/4] extracting task packages")
    by_shard = collections.defaultdict(set)
    for row in reps.itertuples():
        by_shard[row.shard].add(row.member_prefix)
    for shard, prefixes in sorted(by_shard.items()):
        with tarfile.open(dest / shard) as tf:
            members = [m for m in tf if "/".join(m.name.split("/")[:2]) in prefixes]
            tf.extractall(tasks_dir, members=members, filter="data")
        print(f"      {shard}: {len(prefixes)}")

    print("[4/4] writing group manifests (excluding multi-container tasks)")
    root = tasks_dir / "tasks"
    dropped = 0
    for group in range(1, groups + 1):
        rows = []
        for row in reps[reps["group"] == group].itertuples():
            task = root / row.task_id
            if any(
                (task / name).is_file() or (task / "environment" / name).is_file()
                for name in COMPOSE_NAMES
            ):
                dropped += 1
                continue
            rows.append(
                {
                    "task_id": row.task_id,
                    "group": group,
                    "nel": int(row.nel),
                    "category": row.category,
                }
            )
        path = out_root / f"g{group:02d}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        span = (rows[0]["nel"], rows[-1]["nel"]) if rows else (0, 0)
        print(f"      g{group:02d}: {len(rows):4d} tasks   solve.sh lines {span[0]}-{span[1]}")
    print(f"      excluded {dropped} multi-container tasks")

    combined = out_root / "all.jsonl"
    combined.write_text(
        "".join(
            line
            for group in range(1, groups + 1)
            for line in (out_root / f"g{group:02d}.jsonl").read_text().splitlines(keepends=True)
        )
    )
    print(f"\ntasks_root: {root}")
    print(f"manifests : {out_root}/g01.jsonl .. g{groups:02d}.jsonl  (and all.jsonl)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True, help="Where to download the dataset.")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the working set.")
    parser.add_argument("--total", type=int, default=5000, help="How many tasks to select.")
    parser.add_argument("--groups", type=int, default=10, help="How many difficulty bands.")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    dest = args.dest.expanduser()
    out_root = (args.out or dest / "selection").expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download(dest)
    reps = select(dest, args.total, args.groups)
    extract(dest, reps, out_root, args.groups)


if __name__ == "__main__":
    main()
