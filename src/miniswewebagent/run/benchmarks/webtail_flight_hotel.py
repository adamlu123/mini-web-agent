"""Compatibility alias for the reorganized WebTail benchmark module."""

from __future__ import annotations

import sys

from miniswewebagent.run.benchmarks.WTB import webtail_flight_hotel as _implementation

# Return the implementation module itself so monkeypatching this historical
# import path updates the globals used by its worker functions.
sys.modules[__name__] = _implementation
