"""Test harness for `final_script.py`'s `select_depart_return_date` async function.

This CLI is the ONLY thing that should drive `final_script.py` during phase 1
of the datepicker workflow. It keeps `final_script.py` pure and production-ready
(no sessions, no screenshots, no disk I/O inside the async function) while still
producing the per-test-case artefacts (pre/post screenshots + log) that the
oracle judge needs.

For each invocation it:

    1. Validates --departure-date / --return-date (regex + Gregorian + return
       strictly later than departure).
    2. Creates a fresh, dedicated Browserbase session (keepAlive=False; the
       session is bound to this single test case so parallel invocations do
       NOT clobber each other).
    3. Connects via Playwright `connect_over_cdp`, navigates to --start-url,
       sets a 1280x1800 viewport, and saves a `00_pre_*.png` screenshot.
    4. Imports `select_depart_return_date` from the supplied --final-script,
       awaits it on the live page with the two ISO dates.
    5. Saves a `99_post_*.png` screenshot, plus best-effort intermediate
       screenshots of the populated departure-/return-date inputs and the
       reopened calendar so the oracle judge has visual evidence.
    6. Writes `<out-dir>/final_script_log.txt` with the canonical schema
       (`step 0 params: ...` header line + `final_response: ...` footer).
    7. Releases the Browserbase session and disconnects.

Usage:
    python -m miniswewebagent.tools.test_date_picker \\
        --workspace-dir "$WS" \\
        --final-script "$WS/final_script.py" \\
        --start-url https://www.united.com/ \\
        --departure-date 2026-07-10 \\
        --return-date 2026-07-14 \\
        --out-dir "$WS/final_runs/run_001/test_case_1"

Exit code is 0 iff `select_depart_return_date(...)["verified"]` is truthy.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "https://api.browserbase.com/v1/sessions"
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve_path(path_str: str, workspace_dir: str = "") -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        base = Path(workspace_dir) if workspace_dir else Path.cwd()
        path = base / path
    return path


def _require_env(name: str, override: str = "") -> str:
    value = override or os.environ.get(name, "")
    if not value:
        raise SystemExit(f"error: missing required env var {name} (and no --{name.lower().replace('_', '-')} override)")
    return value


def _parse_date(label: str, value: str) -> _dt.date:
    if not _DATE_RE.match(value):
        raise SystemExit(f"error: --{label} {value!r} does not match YYYY-MM-DD")
    try:
        return _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"error: --{label} {value!r} is not a valid Gregorian date: {exc}") from None


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


def _create_session(api_url: str, api_key: str, project_id: str, region: str, retries: int) -> dict[str, str]:
    payload: dict[str, Any] = {
        "projectId": project_id,
        "proxies": True,
        "keepAlive": False,
        "browserSettings": {"advancedStealth": True},
        "timeout": 1200,
    }
    if region:
        payload["region"] = region
    response = _request_with_retry(
        "POST",
        api_url,
        headers={"x-bb-api-key": api_key, "Content-Type": "application/json"},
        json_body=payload,
        retries=retries,
    )
    body = response.json()
    session_id = body.get("id", "")
    connect_url = body.get("connectUrl", "")
    if not session_id or not connect_url:
        raise SystemExit(f"error: Browserbase response missing id/connectUrl: {body!r}")
    return {"id": session_id, "connectUrl": connect_url}


def _release_session(api_url: str, session_id: str, api_key: str, project_id: str, retries: int) -> None:
    try:
        _request_with_retry(
            "POST",
            f"{api_url.rstrip('/')}/{session_id}",
            headers={"x-bb-api-key": api_key, "Content-Type": "application/json"},
            json_body={"projectId": project_id, "status": "REQUEST_RELEASE"},
            retries=retries,
        )
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        print(f"BB_RELEASE_FAILED id={session_id} err={exc}", file=sys.stderr)


def _import_final_script(final_script_path: Path):
    if not final_script_path.exists():
        raise SystemExit(f"error: --final-script not found: {final_script_path}")
    spec = importlib.util.spec_from_file_location("agent_final_script", final_script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: failed to build import spec for {final_script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_final_script"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "select_depart_return_date"):
        raise SystemExit(
            f"error: {final_script_path} does not define `select_depart_return_date` "
            "(expected `async def select_depart_return_date(page, departure_date_iso, return_date_iso)`)"
        )
    return module.select_depart_return_date


async def _run(
    *,
    final_script: Path,
    start_url: str,
    departure_date: str,
    return_date: str,
    out_dir: Path,
    bb_api_url: str,
    bb_api_key: str,
    bb_project_id: str,
    bb_region: str,
    retries: int,
) -> int:
    from playwright.async_api import async_playwright  # local import: optional in some envs

    select_depart_return_date = _import_final_script(final_script)

    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "final_script_log.txt"
    log_lines: list[str] = [
        f"step 0 params: departure_date={departure_date} return_date={return_date}",
    ]

    def _log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    session = _create_session(bb_api_url, bb_api_key, bb_project_id, bb_region, retries)
    _log(f"step 1 action: created Browserbase session {session['id']}")

    verified = False
    result_dict: dict[str, Any] = {}
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.connect_over_cdp(session["connectUrl"])
            try:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
                await page.set_viewport_size({"width": 1280, "height": 1800})

                await page.goto(start_url, wait_until="domcontentloaded")
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                title = await page.title()
                _log(f"step 2 action: loaded start URL {page.url} with title {title!r}")
                pre_path = screenshots_dir / "00_pre_navigated.png"
                await page.screenshot(path=str(pre_path))
                _log(f"step 3 action: saved pre-screenshot {pre_path.name}")

                _log(
                    "step 4 action: invoking select_depart_return_date("
                    f"departure={departure_date}, return={return_date})"
                )
                result_dict = await select_depart_return_date(page, departure_date, return_date)
                if not isinstance(result_dict, dict):
                    raise SystemExit(
                        "error: select_depart_return_date must return a dict; got "
                        f"{type(result_dict).__name__}"
                    )
                verified = bool(result_dict.get("verified"))
                _log(
                    "step 5 action: select_depart_return_date returned "
                    f"verified={verified} departure_selected_label="
                    f"{result_dict.get('departure_selected_label')!r} return_selected_label="
                    f"{result_dict.get('return_selected_label')!r} departure_input_value="
                    f"{result_dict.get('departure_input_value')!r} return_input_value="
                    f"{result_dict.get('return_input_value')!r}"
                )

                post_path = screenshots_dir / "99_post_select.png"
                await page.screenshot(path=str(post_path))
                _log(f"step 6 action: saved post-screenshot {post_path.name}")
            finally:
                try:
                    await browser.close()
                except Exception:
                    try:
                        await browser.disconnect()
                    except Exception:
                        pass
    except Exception:
        tb = traceback.format_exc()
        _log("step ERR action: select_depart_return_date raised; see traceback in run.stderr")
        print(tb, file=sys.stderr)
        verified = False
    finally:
        _release_session(bb_api_url, session["id"], bb_api_key, bb_project_id, retries)
        _log(f"step 7 action: released Browserbase session {session['id']}")

    final_response = (
        f"final_response: departure_date={departure_date} return_date={return_date} "
        f"departure_selected_label={result_dict.get('departure_selected_label')} "
        f"return_selected_label={result_dict.get('return_selected_label')} "
        f"departure_input_value={result_dict.get('departure_input_value')} "
        f"return_input_value={result_dict.get('return_input_value')} "
        f"verified={verified}"
    )
    log_lines.append(final_response)
    print(final_response, flush=True)

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "departure_date": departure_date,
                "return_date": return_date,
                "verified": verified,
                **{k: result_dict.get(k) for k in (
                    "departure_selected_label",
                    "return_selected_label",
                    "departure_input_value",
                    "return_input_value",
                )},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if verified else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m miniswewebagent.tools.test_date_picker",
        description=(
            "Drive `select_depart_return_date(page, departure_date_iso, return_date_iso)` "
            "from a final_script.py against a fresh Browserbase session, capturing "
            "pre/post screenshots and a canonical final_script_log.txt."
        ),
    )
    parser.add_argument("--workspace-dir", default="", help="Base dir for relative paths.")
    parser.add_argument(
        "--final-script",
        required=True,
        help="Path to final_script.py exposing async select_depart_return_date(page, dep, ret).",
    )
    parser.add_argument("--start-url", required=True, help="Flight site URL to load before the call.")
    parser.add_argument("--departure-date", required=True, help="Outbound date (YYYY-MM-DD).")
    parser.add_argument("--return-date", required=True, help="Inbound date (YYYY-MM-DD).")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Per-test-case output dir; receives screenshots/, final_script_log.txt, result.json.",
    )
    parser.add_argument("--api-key", default="", help="Override BROWSERBASE_API_KEY env var.")
    parser.add_argument("--project-id", default="", help="Override BROWSERBASE_PROJECT_ID env var.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Browserbase sessions API base URL.")
    parser.add_argument("--region", default="", help="Optional Browserbase region.")
    parser.add_argument("--retries", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    dep = _parse_date("departure-date", args.departure_date)
    ret = _parse_date("return-date", args.return_date)
    if not (ret > dep):
        raise SystemExit(
            f"error: --return-date ({args.return_date}) must be strictly later than "
            f"--departure-date ({args.departure_date})"
        )

    api_key = _require_env("BROWSERBASE_API_KEY", args.api_key)
    project_id = _require_env("BROWSERBASE_PROJECT_ID", args.project_id)

    final_script = _resolve_path(args.final_script, args.workspace_dir)
    out_dir = _resolve_path(args.out_dir, args.workspace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return asyncio.run(
        _run(
            final_script=final_script,
            start_url=args.start_url,
            departure_date=args.departure_date,
            return_date=args.return_date,
            out_dir=out_dir,
            bb_api_url=args.api_url,
            bb_api_key=api_key,
            bb_project_id=project_id,
            bb_region=args.region,
            retries=args.retries,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
