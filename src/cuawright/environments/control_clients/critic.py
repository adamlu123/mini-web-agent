from __future__ import annotations

import argparse
import json


def request_text(value: str) -> str:
    if len(value) > 2_000:
        raise argparse.ArgumentTypeError("must contain at most 2000 characters")
    return value


def bounded_steps(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer from 1 to 32") from None
    if not 1 <= parsed <= 32:
        raise argparse.ArgumentTypeError("must be an integer from 1 to 32")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True, type=request_text)
parser.add_argument("--max-steps", required=True, type=bounded_steps)
arguments = parser.parse_args()
print(
    json.dumps(
        {
            "cuawright_control": 1,
            "operation": "critic",
            "request": arguments.request,
            "max_steps": arguments.max_steps,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
