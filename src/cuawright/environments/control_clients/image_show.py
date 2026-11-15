from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath


def absolute_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("//")
        or not path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise argparse.ArgumentTypeError("must be a canonical absolute POSIX path")
    return value


parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, type=absolute_path)
arguments = parser.parse_args()
print(
    json.dumps(
        {
            "cuawright_control": 1,
            "operation": "image_show",
            "path": arguments.path,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
