from __future__ import annotations

import argparse
import json


parser = argparse.ArgumentParser()
parser.parse_args()
print(
    json.dumps(
        {"cuawright_control": 1, "operation": "submit"},
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
