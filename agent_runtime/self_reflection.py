"""Neutral, harness-agnostic entry point for the self-reflection judge tool.

Lets agent trajectories invoke the tool as ``python -m self_reflection ...``
instead of ``python -m miniswewebagent.tools.self_reflection ...``, so the
recorded commands do not reference the internal harness package. Behaviour and
CLI flags are identical; the real implementation is simply re-exported.
"""

from __future__ import annotations

import sys

from miniswewebagent.tools.self_reflection import main  # noqa: F401

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
