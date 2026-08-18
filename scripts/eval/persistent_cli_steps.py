#!/usr/bin/env python3
"""Evaluate stored step scripts or result.json action histories."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for import_root in (SRC_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from miniswewebagent.evaluation.om2w.runner import layout_main as main

if __name__ == "__main__":
    main()
