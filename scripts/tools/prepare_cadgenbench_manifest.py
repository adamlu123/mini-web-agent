#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cuawright.environments.cadgenbench.task import (
    DEFAULT_BENCHMARK_COMMIT,
    DEFAULT_DATASET_REVISION,
    discover_tasks,
    resolve_benchmark_commit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic local CadGenBench panel manifest"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cadgenbench-repo",
        default=str(REPO_ROOT / "cache" / "cadgenbench"),
    )
    parser.add_argument(
        "--benchmark-commit",
        default=DEFAULT_BENCHMARK_COMMIT,
    )
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_DATASET_REVISION,
    )
    parser.add_argument("--fixture-id", action="append", dest="fixture_ids", default=[])
    parser.add_argument("--fixture-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _load_fixture_ids(path: str | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in Path(path)
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    benchmark_commit = resolve_benchmark_commit(
        args.cadgenbench_repo,
        expected_commit=args.benchmark_commit,
    )
    fixture_ids = [*args.fixture_ids, *_load_fixture_ids(args.fixture_ids_file)]
    manifest = discover_tasks(
        dataset_dir=args.dataset_dir,
        benchmark_commit=benchmark_commit,
        dataset_revision=args.dataset_revision,
        fixture_ids=fixture_ids or None,
        limit=args.limit,
    )
    payload = {
        "benchmark": "cadgenbench",
        "dataset_dir": str(manifest.dataset_dir),
        "benchmark_commit": manifest.benchmark_commit,
        "dataset_revision": manifest.dataset_revision,
        "tasks": [task.to_manifest_entry() for task in manifest.tasks],
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
