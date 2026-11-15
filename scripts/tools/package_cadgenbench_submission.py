#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cuawright.environments.cadgenbench.packaging import package_submission
from cuawright.environments.cadgenbench.task import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package CadGenBench harness run directories into submission.zip"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--submitter", default=None)
    parser.add_argument(
        "--name", dest="submission_name", default="CUAWright CadGenBench"
    )
    parser.add_argument("--agent-url", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--agree", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    meta = {
        "submitter_name": args.submitter or getpass.getuser(),
        "submission_name": args.submission_name,
        "agent_url": args.agent_url,
        "notes": args.notes,
        "agree_to_publish": bool(args.agree),
    }
    output = package_submission(
        manifest=manifest,
        run_dirs=args.run_dir,
        output_path=args.output,
        meta=meta,
    )
    print(output)


if __name__ == "__main__":
    main()
