#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`scripts.eval.persistent_cli_steps`."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for import_root in (SRC_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# Re-export the legacy module surface for callers that imported this script.
from miniswewebagent.evaluation.om2w.artifacts import *
from miniswewebagent.evaluation.om2w.artifacts import main

if __name__ == "__main__":
    main()
