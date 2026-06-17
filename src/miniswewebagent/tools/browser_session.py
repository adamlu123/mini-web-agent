"""Backend-agnostic browser-session helper for Playwright agent steps.

This module hides *how* a CDP-attachable browser is created so that the
generated trajectory data never contains backend-specific boilerplate
(e.g. the Browserbase REST `create_browserbase_session` block). Every
Playwright step simply does::

    from miniswewebagent.tools.browser_session import open_browser_session

    async with async_playwright() as playwright:
        browser = await open_browser_session(playwright)
        try:
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()
            ...
        finally:
            await browser.close()

The concrete backend is chosen at runtime via the ``MWA_BROWSER_BACKEND``
environment variable (default ``browserbase``) so different users can swap
in their own session provider WITHOUT changing a single line of the agent's
Playwright code — keeping the recorded trajectory agnostic:

    * ``browserbase`` — create a fresh Browserbase cloud session and connect
      over CDP. Requires ``BROWSERBASE_API_KEY`` / ``BROWSERBASE_PROJECT_ID``.
    * ``local``       — launch a Playwright-bundled headless Chromium locally
      and connect over CDP (no cloud account needed).
    * ``cdp``         — connect to an already-running browser whose CDP / ws
      endpoint the user supplies via ``MWA_BROWSER_CDP_URL``. This is the
      fully provider-agnostic escape hatch: the caller owns the browser
      lifecycle and we only attach to it.

A thin CLI mirrors the helper for shell-only diagnostics::

    python -m miniswewebagent.tools.browser_session create   # prints connectUrl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

DEFAULT_BACKEND = "browserbase"
_BROWSERBASE_API_URL = "https://api.browserbase.com/v1/sessions"
_SESSION_LOG_NAME = "browser_sessions.jsonl"


def _session_log_path() -> str | None:
    """Return the JSONL path where opened sessions are recorded, or ``None``.

    Honors ``MWA_BROWSER_SESSION_LOG`` for an explicit path; otherwise places
    the log inside the task workspace exposed to agent steps via
    ``WORKSPACE_DIR``. Returns ``None`` when neither is available so recording
    silently no-ops outside a managed run.
    """
    explicit = os.environ.get("MWA_BROWSER_SESSION_LOG")
    if explicit:
        return explicit
    workspace = os.environ.get("WORKSPACE_DIR")
    if workspace:
        return os.path.join(workspace, _SESSION_LOG_NAME)
    return None


def _record_session(meta: dict[str, Any]) -> None:
    """Append one JSON line describing a freshly opened session to the task dir.

    Best-effort: any I/O error is swallowed so recording never breaks the
    agent's Playwright step. The line is a dict like
    ``{"backend": "browserbase", "id": "...", "ts": 1234567890.0}``.
    """
    path = _session_log_path()
    backend = meta.get("backend")
    if not path or not backend:
        return
    record: dict[str, Any] = {"backend": backend, "ts": time.time()}
    if meta.get("id"):
        record["id"] = meta["id"]
    if meta.get("pid"):
        record["pid"] = meta["pid"]
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _resolve_backend(backend: str | None) -> str:
    value = (backend or os.environ.get("MWA_BROWSER_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if value not in {"browserbase", "local", "cdp"}:
        raise ValueError(
            f"unknown browser backend {value!r}; expected one of browserbase|local|cdp"
        )
    return value


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"missing required env var {name} for the selected browser backend")
    return value


async def _create_connect_url(
    backend: str, playwright: Any | None = None
) -> tuple[str, dict[str, Any]]:
    """Return ``(connectUrl, metadata)`` for the chosen backend."""
    if backend == "cdp":
        url = _require_env("MWA_BROWSER_CDP_URL")
        return url, {"backend": "cdp"}

    if backend == "browserbase":
        import httpx

        api_key = _require_env("BROWSERBASE_API_KEY")
        project_id = _require_env("BROWSERBASE_PROJECT_ID")
        timeout = int(os.environ.get("MWA_BROWSERBASE_TIMEOUT", "720"))
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                os.environ.get("MWA_BROWSERBASE_API_URL", _BROWSERBASE_API_URL),
                headers={"x-bb-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "projectId": project_id,
                    "proxies": True,
                    "browserSettings": {"advancedStealth": True},
                    "timeout": timeout,
                },
            )
            response.raise_for_status()
            body = response.json()
        connect_url = body.get("connectUrl", "")
        if not connect_url:
            raise RuntimeError(f"Browserbase response missing connectUrl: {body!r}")
        return connect_url, {"backend": "browserbase", "id": body.get("id", "")}

    # backend == "local": reuse the local-chromium spawner so the same code
    # path is shared with the persistent local-browser tool.
    from miniswewebagent.tools import local_browser_session as lbs

    if playwright is not None:
        chromium = playwright.chromium.executable_path
    else:
        chromium = lbs._chromium_executable()  # noqa: SLF001 - intentional reuse
    import subprocess

    width = int(os.environ.get("MWA_BROWSER_WIDTH", "1280"))
    height = int(os.environ.get("MWA_BROWSER_HEIGHT", "1800"))
    args = [
        chromium,
        "--remote-debugging-port=0",
        "--headless=new",
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=TranslateUI,MediaRouter",
        f"--window-size={width},{height}",
    ]
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "bufsize": 1,
        "close_fds": True,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(args, **popen_kwargs)  # noqa: S603
    connect_url = lbs._wait_for_devtools_url(proc, 30.0)  # noqa: SLF001 - intentional reuse
    return connect_url, {"backend": "local", "pid": proc.pid}


async def open_browser_session(playwright: Any, *, backend: str | None = None) -> Any:
    """Create/attach a browser via the configured backend and return it connected.

    The returned object is a Playwright ``Browser`` already attached over CDP.
    Callers obtain a context/page from it as usual and should end with
    ``await browser.close()`` (for cloud/local backends this also ends the
    fresh session; for the ``cdp`` backend it only detaches the connection).

    Each opened session is also appended (one JSON line) to the task's
    ``browser_sessions.jsonl`` log so the run can release any cloud session
    afterwards via :func:`release_recorded_sessions`, even if a step crashes
    before ``browser.close()`` runs.
    """
    resolved = _resolve_backend(backend)
    connect_url, meta = await _create_connect_url(resolved, playwright)
    _record_session(meta)
    return await playwright.chromium.connect_over_cdp(connect_url)


def _read_session_log(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


async def _release_browserbase_session(client: Any, session_id: str) -> tuple[str, bool, str]:
    api_key = os.environ.get("BROWSERBASE_API_KEY", "")
    project_id = os.environ.get("BROWSERBASE_PROJECT_ID", "")
    if not api_key or not project_id:
        return session_id, False, "missing BROWSERBASE_API_KEY/BROWSERBASE_PROJECT_ID"
    base = os.environ.get("MWA_BROWSERBASE_API_URL", _BROWSERBASE_API_URL)
    try:
        resp = await client.post(
            f"{base}/{session_id}",
            headers={"x-bb-api-key": api_key, "Content-Type": "application/json"},
            json={"projectId": project_id, "status": "REQUEST_RELEASE"},
        )
    except Exception as exc:  # network error, etc.
        return session_id, False, str(exc)
    if resp.status_code < 400:
        return session_id, True, ""
    return session_id, False, f"HTTP {resp.status_code}: {resp.text[:200]}"


async def _release_recorded_sessions_async(records: list[dict[str, Any]]) -> list[tuple[str, bool, str]]:
    import httpx

    results: list[tuple[str, bool, str]] = []
    seen: set[str] = set()
    async with httpx.AsyncClient(timeout=30) as client:
        for record in records:
            if record.get("backend") != "browserbase":
                continue
            session_id = str(record.get("id") or "")
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            results.append(await _release_browserbase_session(client, session_id))
    return results


def release_recorded_sessions(
    log_path: str | os.PathLike[str] | None = None,
) -> list[tuple[str, bool, str]]:
    """Release every cloud session recorded by :func:`open_browser_session`.

    Reads the JSONL log written during the run and asks the provider
    (currently Browserbase, via ``POST /v1/sessions/{id}`` with
    ``status=REQUEST_RELEASE``) to end each session. This guarantees cleanup
    even when an agent step aborted before reaching ``await browser.close()``,
    so a cloud session is not left running until its server-side ``timeout``.

    Best-effort and idempotent: returns a list of ``(session_id, ok, error)``
    tuples; releasing an already-ended session simply succeeds.
    """
    path = str(log_path) if log_path else _session_log_path()
    if not path or not os.path.exists(path):
        return []
    records = _read_session_log(path)
    if not records:
        return []
    return asyncio.run(_release_recorded_sessions_async(records))


async def _cmd_create(args: argparse.Namespace) -> int:
    backend = _resolve_backend(args.backend)
    connect_url, meta = await _create_connect_url(backend)
    print(f"BROWSER_BACKEND={meta.get('backend', backend)}")
    print(f"BROWSER_CONNECT_URL={connect_url}")
    if meta.get("id"):
        print(f"BROWSER_SESSION_ID={meta['id']}")
    if meta.get("pid"):
        print(f"BROWSER_SESSION_PID={meta['pid']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m miniswewebagent.tools.browser_session",
        description="Backend-agnostic browser-session helper (browserbase|local|cdp).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Create/attach a session and print its connectUrl.")
    create.add_argument(
        "--backend",
        default="",
        help="Override MWA_BROWSER_BACKEND (browserbase|local|cdp).",
    )
    create.set_defaults(func=_cmd_create)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
