#!/usr/bin/env python3
"""Evaluate persistent-browser actions and root-level screenshots."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for import_root in (SRC_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from miniswewebagent.evaluation.om2w.runner import main

if __name__ == "__main__":
    main()
