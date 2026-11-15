from __future__ import annotations

from typing import Any

from cuawright.environments.desktop.guest_tools import (
    _PASSWORD_SUDO_WRAPPER as DESKTOP_PASSWORD_SUDO_WRAPPER,
)
from cuawright.environments.desktop.guest_tools import (
    control_clients_ready as _control_clients_ready,
)
from cuawright.environments.desktop.guest_tools import (
    provision_control_clients as _provision_control_clients,
)
from cuawright.environments.desktop.guest_tools import (
    write_guest_file as _write_guest_file,
)
from cuawright.environments.desktop.knowledge import (
    DESKTOP_KNOWLEDGE_SCRIPT as OSW_SCRIPT,
)

_PASSWORD_SUDO_WRAPPER = DESKTOP_PASSWORD_SUDO_WRAPPER


KNOWLEDGE_FILES = {
    "browser.md": """# Chrome and Playwright

## Inspect the existing session first

OSWorld task setup normally opens the required site, account, browser profile,
and task tab before the agent starts. Inspect and preserve that state before
launching or restarting Chrome:

```bash
curl -fsS http://127.0.0.1:1337/json/version
curl -fsS http://127.0.0.1:1337/json/list
pgrep -af 'google-chrome|chromium'
```

Port 1337 is the normal guest-local Chrome debugging endpoint. Some task setups
also expose a guest-local socat mirror on port 9222. Probe it only when 1337 is
unavailable:

```bash
curl -fsS http://127.0.0.1:9222/json/version
```

## Attach with the provisioned Playwright interpreter

OSWorld provisions Playwright at
`/home/user/.local/share/osw/venv/bin/python`. Set the endpoint explicitly for
each script:

```bash
cd "$HOME"
OSW_CDP_URL=http://127.0.0.1:1337 \
  /home/user/.local/share/osw/venv/bin/python /tmp/browser_task.py
```

Enumerate the existing pages and select the task page from its URL or title.
Do not assume the first page is the intended tab:

```python
import os
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp(os.environ["OSW_CDP_URL"])
    pages = [
        page
        for context in browser.contexts
        for page in context.pages
    ]
    for index, page in enumerate(pages):
        print(index, page.url, page.title())
    page = next(
        (page for page in pages if "task-derived-host" in page.url),
        None,
    )
    if page is None:
        raise RuntimeError("Task page not found in the existing Chrome session")
    page.bring_to_front()
```

Prefer real navigation and locator-driven clicks, typing, uploads, and
readbacks on the foreground task page. Use `page.bring_to_front()` before
important visual actions. Do not substitute background HTTP fetches, hidden
tabs, headless requests, or bulk `page.content()` scraping for changing and
confirming the visible desktop state.

Prefer `get_by_role`, accessible names, labels, and the browser accessibility
tree over brittle raw DOM selectors. When CDP is allowed, inspect the Chromium
AXTree without mutating the page:

```python
session = page.context.new_cdp_session(page)
nodes = session.send("Accessibility.getFullAXTree")["nodes"]
for node in nodes:
    role = node.get("role", {}).get("value", "")
    name = node.get("name", {}).get("value", "")
    if role or name:
        print(role, repr(name))
```

DOM inspection may support element discovery, but use the visible application
rather than hidden backend APIs or databases.

## Recover only when the normal session is unavailable

If Chrome is running but neither endpoint is usable, inspect the live window
and process state before restarting anything. Prefer visible UI control when it
preserves valid task state.

If Chrome is genuinely absent, launch it on the real display and verify the
resulting profile and page before continuing:

```bash
export DISPLAY=:0
nohup google-chrome --remote-debugging-port=1337 \
  --no-first-run --no-default-browser-check \
  >/tmp/chrome.log 2>&1 &
curl -fsS http://127.0.0.1:1337/json/version
```

Start the optional 9222 mirror only if a tool specifically needs it:

```bash
pgrep -f 'socat tcp-listen:9222' >/dev/null ||
  nohup socat tcp-listen:9222,fork tcp:localhost:1337 \
  >/tmp/socat.log 2>&1 &
```

If the task forbids CDP or developer tools, do not probe or attach through CDP;
use screenshots, accessibility, keyboard, and visible application controls.

## Troubleshooting

If the provisioned interpreter cannot import Playwright, repair only that
environment:

```bash
python3 -m venv "$HOME/.local/share/osw/venv"
"$HOME/.local/share/osw/venv/bin/pip" install --quiet \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org playwright
```

Do not name Python scripts after standard-library or common package modules such
as `inspect.py`, `json.py`, `typing.py`, `asyncio.py`, or `playwright.py`.
""",
    "native-apps.md": """# Native desktop applications

Use ordinary Python with pyatspi or pyautogui when native UI interaction is
necessary. Capture a screenshot with:

```bash
DISPLAY=:0 python3 - <<'PY'
import pyautogui
pyautogui.screenshot('/tmp/screen.png')
PY
```

Inspect the accessibility tree with:

```bash
DISPLAY=:0 python3 - <<'PY'
import pyatspi
desktop = pyatspi.Registry.getDesktop(0)
stack = [(desktop, 0)]
while stack:
    node, depth = stack.pop()
    try:
        print('  ' * depth, node.getRoleName(), repr(node.name))
        children = list(node)
    except Exception:
        children = []
    stack.extend((child, depth + 1) for child in reversed(children))
PY
```

If Pillow, document libraries, or command-line tools edit an artifact while
GIMP, LibreOffice, WPS, or another editor already has it open, close and reopen
the saved path when application-visible state or evaluator save behavior may
matter. Explicitly save before submission.

Opening or previewing an artifact does not prove a native application imported
it. Use the application's normal import flow and verify persistent content
after reopening the destination application.
""",
}


def provision_guest_tools(
    controller: Any,
    setup_controller: Any | None = None,
) -> None:
    if not _control_clients_ready(controller):
        _provision_control_clients(controller, setup_controller)
    _write_guest_file(
        controller,
        "/home/user/.local/bin/osw",
        OSW_SCRIPT.encode("utf-8"),
        0o755,
    )
    for name, content in KNOWLEDGE_FILES.items():
        _write_guest_file(
            controller,
            f"/home/user/.local/share/osw/kb/{name}",
            content.encode("utf-8"),
            0o644,
        )
    result = controller.run_bash_script(
        """
        set -e
        root="$HOME/.local/share/osw"
        if [ ! -x "$root/venv/bin/python" ]; then
          python3 -m venv "$root/venv"
        fi
        if ! "$root/venv/bin/python" -c 'import playwright' >/dev/null 2>&1; then
          "$root/venv/bin/pip" install --quiet \
            --trusted-host pypi.org \
            --trusted-host files.pythonhosted.org \
            playwright
        fi
        if ! python3 -c 'import openpyxl, docx, pypdf, PIL, requests, bs4, lxml' >/dev/null 2>&1; then
          python3 -m pip install --user --quiet \
            --trusted-host pypi.org \
            --trusted-host files.pythonhosted.org \
            openpyxl python-docx pypdf pillow requests beautifulsoup4 lxml
        fi
        """,
        timeout=240,
    )
    if not result or result.get("status") != "success" or result.get("returncode") != 0:
        raise RuntimeError(f"Failed to provision guest dependencies: {result}")
