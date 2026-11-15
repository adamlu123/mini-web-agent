from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuawright.environments.cadgenbench.container_tools import (
    public_evaluate_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = public_evaluate_candidate(args.candidate, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
