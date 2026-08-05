#!/usr/bin/env python3
"""Compatibility entry point for the reorganized Qwen3.5 preprocessor."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CANONICAL_SCRIPT = _SCRIPT_DIR / "data/qwen35/preprocess_lastobs_singleturn.py"

if __name__ == "__main__":
    runpy.run_path(str(_CANONICAL_SCRIPT), run_name="__main__")
else:
    _repo_root = str(_SCRIPT_DIR.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from scripts.data.qwen35.preprocess_lastobs_singleturn import *
