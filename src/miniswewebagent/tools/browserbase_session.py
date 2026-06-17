"""CLI for managing a long-lived, reusable Browserbase cloud session.

Lets the agent (or any bash step) attach to ONE persistent Browserbase
session across multiple processes by:

    * `create`  -> POST /v1/sessions with keepAlive=True, write
                   `{id, connectUrl}` to a JSON file on disk, print the id.
    * `info`    -> GET  /v1/sessions/<id>, print the JSON status.
    * `release` -> POST /v1/sessions/<id> with status=REQUEST_RELEASE.

Subsequent Playwright processes load the JSON file, call
`playwright.chromium.connect_over_cdp(connectUrl)`, and `browser.disconnect()`
(NEVER `browser.close()`) so the session keeps living for the next step.

Usage:
    python -m miniswewebagent.tools.browserbase_session create  --out .bb_session.json
    python -m miniswewebagent.tools.browserbase_session info    --session-file .bb_session.json
    python -m miniswewebagent.tools.browserbase_session release --session-file .bb_session.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "https://api.browserbase.com/v1/sessions"
DEFAULT_SESSION_FILE = ".bb_session.json"

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _resolve_path(path_str: str, workspace_dir: str = "") -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        base = Path(workspace_dir) if workspace_dir else Path.cwd()
        path = base / path
    return path


def _require_env(name: str, override: str = "") -> str:
    value = override or os.environ.get(name, "")
    if not value:
        raise SystemExit(f"error: missing required env var {name} (and no --{name.lower()} override)")
    return value


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    retries: int = 4,
    backoff_seconds: float = 1.0,
    timeout: float = 30.0,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.request(method, url, headers=headers, json=json_body)
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < retries:
                time.sleep(min(backoff_seconds * (2 ** (attempt - 1)), 8.0))
                continue
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(min(backoff_seconds * (2 ** (attempt - 1)), 8.0))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _RETRYABLE_STATUS_CODES and attempt < retries:
                last_exc = exc
                time.sleep(min(backoff_seconds * (2 ** (attempt - 1)), 8.0))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _cmd_create(args: argparse.Namespace) -> int:
    api_key = _require_env("BROWSERBASE_API_KEY", args.api_key)
    project_id = _require_env("BROWSERBASE_PROJECT_ID", args.project_id)

    payload: dict[str, Any] = {
        "projectId": project_id,
        "proxies": args.proxies,
        "keepAlive": True,
        "browserSettings": {"advancedStealth": True},
        "timeout": args.timeout,
    }
    if args.region:
        payload["region"] = args.region

    response = _request_with_retry(
        "POST",
        args.api_url,
        headers={"x-bb-api-key": api_key, "Content-Type": "application/json"},
        json_body=payload,
        retries=args.retries,
    )
    body = response.json()

    session = {
        "id": body.get("id", ""),
        "connectUrl": body.get("connectUrl", ""),
        "projectId": project_id,
        "apiUrl": args.api_url,
        "createdAt": body.get("createdAt", ""),
        "region": body.get("region", args.region or ""),
    }
    if not session["id"] or not session["connectUrl"]:
        raise SystemExit(f"error: Browserbase response missing id/connectUrl: {body!r}")

    out_path = _resolve_path(args.out, args.workspace_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

    print(f"BB_SESSION_ID={session['id']}")
    print(f"BB_SESSION_FILE={out_path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    api_key = _require_env("BROWSERBASE_API_KEY", args.api_key)
    session = json.loads(_resolve_path(args.session_file, args.workspace_dir).read_text(encoding="utf-8"))
    session_id = session["id"]
    api_url = session.get("apiUrl", DEFAULT_API_URL)

    response = _request_with_retry(
        "GET",
        f"{api_url.rstrip('/')}/{session_id}",
        headers={"x-bb-api-key": api_key},
        retries=args.retries,
    )
    print(json.dumps(response.json(), indent=2))
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    api_key = _require_env("BROWSERBASE_API_KEY", args.api_key)
    session_path = _resolve_path(args.session_file, args.workspace_dir)
    if not session_path.exists():
        print(f"BB_RELEASE_SKIPPED missing={session_path}")
        return 0
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session_id = session["id"]
    project_id = session.get("projectId") or _require_env("BROWSERBASE_PROJECT_ID", args.project_id)
    api_url = session.get("apiUrl", DEFAULT_API_URL)

    response = _request_with_retry(
        "POST",
        f"{api_url.rstrip('/')}/{session_id}",
        headers={"x-bb-api-key": api_key, "Content-Type": "application/json"},
        json_body={"projectId": project_id, "status": "REQUEST_RELEASE"},
        retries=args.retries,
    )
    print(f"BB_RELEASE_REQUESTED id={session_id} status_code={response.status_code}")
    if args.delete_file:
        try:
            session_path.unlink()
            print(f"BB_SESSION_FILE_DELETED {session_path}")
        except OSError as exc:
            print(f"BB_SESSION_FILE_DELETE_FAILED {session_path} {exc}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m miniswewebagent.tools.browserbase_session",
        description="Manage a keep-alive Browserbase session shared across bash steps.",
    )
    parser.add_argument(
        "--workspace-dir",
        default="",
        help="Resolve --out / --session-file relative to this directory (defaults to cwd).",
    )
    parser.add_argument("--api-key", default="", help="Override BROWSERBASE_API_KEY env var.")
    parser.add_argument("--project-id", default="", help="Override BROWSERBASE_PROJECT_ID env var.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Browserbase sessions API base URL.")
    parser.add_argument("--retries", type=int, default=4, help="Retry attempts for transient errors.")

    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a keep-alive session and persist its id+connectUrl.")
    create.add_argument("--out", default=DEFAULT_SESSION_FILE, help="Where to write the session JSON.")
    create.add_argument("--proxies", action=argparse.BooleanOptionalAction, default=True)
    create.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Browserbase session lifetime in seconds (max 21600).",
    )
    create.add_argument("--region", default="", help="Optional Browserbase region (e.g. us-east-1).")
    create.set_defaults(func=_cmd_create)

    info = sub.add_parser("info", help="Print Browserbase session status JSON.")
    info.add_argument("--session-file", default=DEFAULT_SESSION_FILE)
    info.set_defaults(func=_cmd_info)

    release = sub.add_parser("release", help="Request release of the persisted session.")
    release.add_argument("--session-file", default=DEFAULT_SESSION_FILE)
    release.add_argument(
        "--delete-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also delete the session JSON file after release.",
    )
    release.set_defaults(func=_cmd_release)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
