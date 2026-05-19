"""Auto-loaded shim that records every Browserbase session creation.

Activated by adding this directory to ``PYTHONPATH``. Writes one JSONL line
per ``POST https://api.browserbase.com/v1/sessions`` call to
``$BROWSERBASE_SESSIONS_LOG``. Safe no-op if the env var is unset.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

_SESSION_URL = "https://api.browserbase.com/v1/sessions"


def _log_path() -> str | None:
    return os.environ.get("BROWSERBASE_SESSIONS_LOG") or None


def _append(record: dict[str, Any]) -> None:
    path = _log_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _record(url: str, status_code: int | None, body_text: str | None) -> None:
    if not isinstance(url, str) or _SESSION_URL not in url:
        return
    session_id = None
    connect_url = None
    if body_text:
        try:
            data = json.loads(body_text)
            if isinstance(data, dict):
                session_id = data.get("id")
                connect_url = data.get("connectUrl")
        except Exception:
            pass
    _append({
        "ts": time.time(),
        "pid": os.getpid(),
        "url": url,
        "status_code": status_code,
        "session_id": session_id,
        "connect_url": connect_url,
    })


# httpx ----------------------------------------------------------------------
try:
    import httpx  # type: ignore

    _orig_client_post = httpx.Client.post
    _orig_async_client_post = httpx.AsyncClient.post
    _orig_module_post = httpx.post

    def _client_post(self, url, *args, **kwargs):  # type: ignore[no-untyped-def]
        resp = _orig_client_post(self, url, *args, **kwargs)
        try:
            _record(str(resp.request.url), resp.status_code, resp.text)
        except Exception:
            pass
        return resp

    async def _async_client_post(self, url, *args, **kwargs):  # type: ignore[no-untyped-def]
        resp = await _orig_async_client_post(self, url, *args, **kwargs)
        try:
            _record(str(resp.request.url), resp.status_code, resp.text)
        except Exception:
            pass
        return resp

    def _module_post(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        resp = _orig_module_post(url, *args, **kwargs)
        try:
            _record(str(resp.request.url), resp.status_code, resp.text)
        except Exception:
            pass
        return resp

    httpx.Client.post = _client_post  # type: ignore[assignment]
    httpx.AsyncClient.post = _async_client_post  # type: ignore[assignment]
    httpx.post = _module_post  # type: ignore[assignment]
except Exception:
    pass


# requests -------------------------------------------------------------------
try:
    import requests  # type: ignore

    _orig_session_post = requests.Session.post
    _orig_requests_post = requests.post

    def _session_post(self, url, *args, **kwargs):  # type: ignore[no-untyped-def]
        resp = _orig_session_post(self, url, *args, **kwargs)
        try:
            _record(str(resp.url), resp.status_code, resp.text)
        except Exception:
            pass
        return resp

    def _requests_post(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        resp = _orig_requests_post(url, *args, **kwargs)
        try:
            _record(str(resp.url), resp.status_code, resp.text)
        except Exception:
            pass
        return resp

    requests.Session.post = _session_post  # type: ignore[assignment]
    requests.post = _requests_post  # type: ignore[assignment]
except Exception:
    pass
