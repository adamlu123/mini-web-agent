from __future__ import annotations

import re
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miniswewebagent import package_dir
from miniswewebagent.run.utilities.trace_viewer import build_server

VIEWER_DIR = package_dir / "viewer"


def _start_server(root: Path):
    server = build_server(root, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_index_html_wires_up_the_compare_tab() -> None:
    html = (VIEWER_DIR / "index.html").read_text(encoding="utf-8")

    assert '<script src="compare.js"></script>' in html
    # Tab bar and section anchors compare.js expects to find via getElementById.
    for element_id in (
        "traceTabBtn",
        "compareTabBtn",
        "traceView",
        "compareView",
        "compareRunA",
        "compareRunB",
        "compareLoadBtn",
        "leaderboardBody",
        "compareFilterChips",
        "compareTaskList",
        "compareColumns",
    ):
        assert f'id="{element_id}"' in html, f"index.html is missing #{element_id}"


def test_compare_js_is_scoped_in_an_iife() -> None:
    """Regression test: app.js and compare.js share one global <script> scope.

    compare.js previously declared top-level fetchJson/escapeHtml/
    populateRunSelect helpers that collided with app.js's same-named
    globals, silently overwriting them (Trace tab broke at runtime with
    "Cannot read properties of undefined (reading 'value')"). compare.js
    must stay wrapped in an IIFE so its declarations cannot leak.
    """
    lines = [line for line in (VIEWER_DIR / "compare.js").read_text(encoding="utf-8").splitlines() if line.strip()]
    non_comment_lines = [line for line in lines if not line.strip().startswith("//")]

    assert non_comment_lines[0].strip() == "(function () {", (
        "compare.js must open with an IIFE so its top-level declarations don't leak "
        "into the global scope shared with app.js"
    )
    assert non_comment_lines[-1].strip() == "})();", "compare.js must close the IIFE it opens with"

    # None of app.js's top-level helper names may reappear as unindented (i.e.
    # top-level, outside the IIFE) declarations in compare.js.
    app_js = (VIEWER_DIR / "app.js").read_text(encoding="utf-8")
    top_level_app_js_names = set(re.findall(r"^(?:function|const) (\w+)", app_js, flags=re.MULTILINE))
    compare_js = (VIEWER_DIR / "compare.js").read_text(encoding="utf-8")
    leaked_top_level_names = set(re.findall(r"^(?:function|const) (\w+)", compare_js, flags=re.MULTILINE))
    assert not (top_level_app_js_names & leaked_top_level_names)


def test_server_serves_compare_js(tmp_path: Path) -> None:
    server = _start_server(tmp_path / "outputs")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/compare.js") as response:
            assert response.status == 200
            assert "javascript" in response.headers.get("Content-Type", "")
            body = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()

    assert body.strip().startswith("//")
    assert "(function () {" in body
