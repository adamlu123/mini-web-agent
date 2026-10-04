"""Neutral entry point for attaching a workspace image to the policy model.

Persistent-browser trajectories invoke ``python -m image_read --path PATH``.
The local workspace broker recognizes that exact standalone command and turns
the validated descriptor into an image part on the next model observation.
"""

from __future__ import annotations

import sys

from miniswewebagent.tools.image_read import main

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
