"""Convert an Online-Mind2Web JSON tasks file into a parquet usable by the
web-agent RL dataset.

Example::

    python -m echo_rl.web_agent.scripts.prepare_om2w_data \\
        --input /home/luyadong/sandbox/nano_eval/task_files/om2w_260220.json \\
        --output /home/luyadong/sandbox/echo-rl/data/web_agent/om2w_train.parquet \\
        --split train --train-size 60 --val-output \\
            /home/luyadong/sandbox/echo-rl/data/web_agent/om2w_val.parquet \\
        --val-size 8

The parquet rows are pre-tokenized at load time inside ``WebAgentTaskDataset``,
so this script only normalizes records and writes plain columns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Allow running this script directly from a checkout without installation.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echo_rl.web_agent.dataset import load_om2w_records  # noqa: E402


def _write_parquet(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty parquet file.")
    columns = {
        "task_id": [str(row["task_id"]) for row in rows],
        "task": [str(row["task"]) for row in rows],
        "start_url": [str(row["start_url"]) for row in rows],
        "level": [str(row.get("level", "")) for row in rows],
        "reference_length": [int(row.get("reference_length", 0)) for row in rows],
    }
    table = pa.table(columns)
    pq.write_table(table, str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an Online-Mind2Web JSON file.")
    parser.add_argument("--output", required=True, help="Output parquet path for train split.")
    parser.add_argument("--val-output", default=None, help="Optional parquet path for val split.")
    parser.add_argument("--train-size", type=int, default=None, help="Cap on the number of training rows.")
    parser.add_argument("--val-size", type=int, default=None, help="Cap on the number of validation rows.")
    parser.add_argument(
        "--levels", nargs="*", default=None,
        help="If set, keep only tasks whose 'level' is in this list (e.g. easy medium).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when sampling rows.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep records in source order instead of shuffling before sampling.",
    )
    args = parser.parse_args()

    records = load_om2w_records(args.input)
    if args.levels:
        wanted = {lvl.lower() for lvl in args.levels}
        records = [r for r in records if str(r.get("level", "")).lower() in wanted]
    if not records:
        raise SystemExit(f"No records survived filtering from {args.input}.")

    if not args.no_shuffle:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(records)

    train_size = args.train_size if args.train_size is not None else len(records)
    train_rows = records[:train_size]

    if args.val_output:
        val_start = len(train_rows)
        val_size = args.val_size if args.val_size is not None else max(1, len(records) - val_start)
        val_rows = records[val_start : val_start + val_size]
        if not val_rows:
            raise SystemExit("No rows left for the val split. Use a smaller --train-size or larger input.")
        _write_parquet(val_rows, Path(args.val_output))
        print(f"Wrote {len(val_rows)} val rows to {args.val_output}")

    _write_parquet(train_rows, Path(args.output))
    print(f"Wrote {len(train_rows)} train rows to {args.output}")
    print(json.dumps({"input": os.path.abspath(args.input), "train_size": len(train_rows)}, indent=2))


if __name__ == "__main__":
    main()
