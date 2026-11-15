"""Add the CUAWright source tree and repository root to ``sys.path``."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
