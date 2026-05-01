#!/usr/bin/env python3
"""Run the verified Google Flights comparison CLI for this skill."""

from pathlib import Path
import os
import runpy
import sys


SKILL_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = Path(
    os.environ.get("GOOGLE_FLIGHTS_SOURCE_SCRIPT", str(SKILL_DIR / "final_script.py"))
).expanduser()


def main() -> None:
    if not SOURCE_SCRIPT.exists():
        raise SystemExit(f"Source script not found: {SOURCE_SCRIPT}")

    os.environ.setdefault("LOCAL_BROWSER_CDP_URL", "http://127.0.0.1:9222")
    os.environ.setdefault("WORKSPACE_DIR", str(SKILL_DIR))

    sys.argv[0] = str(SOURCE_SCRIPT)
    runpy.run_path(str(SOURCE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
