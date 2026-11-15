from __future__ import annotations

import argparse
import json


def question_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be non-empty")
    if len(value) > 2_000:
        raise argparse.ArgumentTypeError("must contain at most 2000 characters")
    return value


parser = argparse.ArgumentParser()
parser.add_argument("--question", required=True, type=question_text)
arguments = parser.parse_args()
print(
    json.dumps(
        {
            "cuawright_control": 1,
            "operation": "ask_user",
            "question": arguments.question,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
