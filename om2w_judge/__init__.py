"""Package compatibility for the verbatim Online-Mind2Web sources."""

import sys
from importlib import import_module

from om2w_judge import utils as _utils

# Official modules expect their flat ``src`` directory on sys.path. Temporarily
# provide that layout while eagerly loading the one module that imports ``utils``;
# restore any pre-existing top-level module immediately afterward.
_MISSING = object()
_previous_utils = sys.modules.get("utils", _MISSING)
sys.modules["utils"] = _utils
try:
    import_module("om2w_judge.methods.webjudge_online_mind2web")
finally:
    if _previous_utils is _MISSING:
        sys.modules.pop("utils", None)
    else:
        sys.modules["utils"] = _previous_utils
