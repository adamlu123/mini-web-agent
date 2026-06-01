"""Neutral, harness-agnostic entry point for the browser-session helper.

This module exists so that generated agent trajectories import the browser
helper under a generic top-level name:

    from browser_session import open_browser_session

instead of referencing any specific agent-harness package. The directory
containing this file is placed on ``PYTHONPATH`` by the agnostic agent config
(see ``best_default_judge_json_agnostic.yaml``), so the import resolves the
same way regardless of which harness produced the trajectory.

The actual implementation (backend selection via ``MWA_BROWSER_BACKEND`` =
browserbase | local | cdp) lives elsewhere and is simply re-exported here; that
internal detail is invisible to the recorded trajectory, which only ever shows
the neutral ``browser_session`` import above.
"""

from __future__ import annotations

from miniswewebagent.tools.browser_session import (  # noqa: F401
    main,
    open_browser_session,
)

__all__ = ["main", "open_browser_session"]


if __name__ == "__main__":
    import sys

    sys.exit(main())
