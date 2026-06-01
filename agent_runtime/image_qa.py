"""Neutral, harness-agnostic entry point for the image-QA tool.

Lets agent trajectories invoke the tool as ``python -m image_qa ...`` instead of
``python -m miniswewebagent.tools.image_qa ...``, so the recorded commands do not
reference the internal harness package. Behaviour and CLI flags are identical; the
real implementation is simply re-exported.
"""

from __future__ import annotations

import sys

from miniswewebagent.tools.image_qa import main  # noqa: F401

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
