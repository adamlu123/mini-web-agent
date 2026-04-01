#!/usr/bin/env python3
"""Test Browserbase session creation + URL navigation for all benchmark URLs.

Usage:
    source ~/cred.sh && python scripts/test_browserbase_urls.py

Concurrency is capped at MAX_CONCURRENT (default 30) to avoid rate limits.
Each session is created, navigated to the URL, checked, then released.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "miniswewebagent",
    "run",
    "benchmarks",
    "om2w_260220.json",
)
MAX_CONCURRENT = 30  # max parallel sessions
NAVIGATION_TIMEOUT_MS = 30_000  # 30s to load a page
SESSION_TIMEOUT = 120  # browserbase keep_alive timeout

BB_API_KEY = os.environ["BROWSERBASE_API_KEY"]
BB_PROJECT_ID = os.environ["BROWSERBASE_PROJECT_ID"]
BB_API_URL = "https://api.browserbase.com/v1/sessions"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CAPTCHA_WAIT_TIMEOUT = 120  # max seconds to wait for captcha resolution


@dataclass
class URLResult:
    url: str
    success: bool = False
    session_id: str = ""
    error: str = ""
    error_type: str = ""
    page_title: str = ""
    elapsed_s: float = 0.0
    captcha_triggered: bool = False
    captcha_resolved: bool = False
    captcha_started_count: int = 0
    captcha_finished_count: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def release_session(session_id: str) -> None:
    """Release a Browserbase session via the REST API."""
    import httpx

    url = f"{BB_API_URL}/{session_id}"
    payload = {"projectId": BB_PROJECT_ID, "status": "REQUEST_RELEASE"}
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            resp = await client.post(
                url, headers={"x-bb-api-key": BB_API_KEY}, json=payload
            )
            resp.raise_for_status()
    except Exception:
        pass  # best-effort cleanup


async def test_single_url(url: str, sem: asyncio.Semaphore) -> URLResult:
    """Create a BB session, connect via Playwright, navigate, and release."""
    from browserbase import Browserbase
    from playwright.async_api import async_playwright

    result = URLResult(url=url)
    t0 = time.monotonic()

    async with sem:
        session_id = None
        try:
            # 1. Create session
            bb = Browserbase(api_key=BB_API_KEY)
            created = await asyncio.to_thread(
                bb.sessions.create,
                project_id=BB_PROJECT_ID,
                proxies=True,
                browser_settings={"advanced_stealth": True},
                keep_alive=True,
                timeout=SESSION_TIMEOUT,
                region="us-east-1",
            )
            session_id = getattr(created, "id", "") or ""
            connect_url = getattr(created, "connect_url", "") or ""
            result.session_id = session_id

            if not connect_url:
                raise RuntimeError("Session created but no connect_url returned")

            # 2. Connect via Playwright CDP
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(
                    connect_url, timeout=30_000
                )
                try:
                    context = browser.contexts[0] if browser.contexts else await browser.new_context()
                    page = context.pages[0] if context.pages else await context.new_page()

                    # Track captcha events via console messages
                    captcha_event = asyncio.Event()
                    captcha_event.set()  # not blocked initially

                    def on_console(msg):
                        if msg.text == "browserbase-solving-started":
                            result.captcha_triggered = True
                            result.captcha_started_count += 1
                            captcha_event.clear()
                        elif msg.text == "browserbase-solving-finished":
                            result.captcha_finished_count += 1
                            captcha_event.set()

                    page.on("console", on_console)

                    # 3. Navigate
                    await page.goto(url, timeout=NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded")

                    # 4. If captcha was triggered, wait for resolution
                    if result.captcha_triggered and not captcha_event.is_set():
                        try:
                            await asyncio.wait_for(
                                captcha_event.wait(),
                                timeout=CAPTCHA_WAIT_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            pass  # captcha never resolved

                    # 5. Quick readiness check
                    try:
                        await page.wait_for_load_state("load", timeout=10_000)
                    except Exception:
                        pass

                    # Check for post-navigation captcha (some sites trigger after load)
                    if not captcha_event.is_set():
                        try:
                            await asyncio.wait_for(
                                captcha_event.wait(),
                                timeout=CAPTCHA_WAIT_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            pass

                    result.captcha_resolved = (
                        result.captcha_started_count > 0
                        and result.captcha_started_count == result.captcha_finished_count
                    )

                    result.page_title = await page.title() or ""
                    result.success = True
                finally:
                    await browser.close()

        except Exception as exc:
            result.error = str(exc)[:500]
            result.error_type = type(exc).__name__
        finally:
            result.elapsed_s = time.monotonic() - t0
            if session_id:
                await release_session(session_id)

    status = "OK" if result.success else "FAIL"
    captcha_info = ""
    if result.captcha_triggered:
        cap_status = "resolved" if result.captcha_resolved else "STUCK"
        captcha_info = f"  [CAPTCHA: {cap_status} started={result.captcha_started_count} finished={result.captcha_finished_count}]"
    print(f"  [{status}] {url}  ({result.elapsed_s:.1f}s){captcha_info}" +
          (f"  -- {result.error_type}: {result.error[:120]}" if result.error else ""))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # Load unique URLs
    with open(os.path.abspath(BENCHMARK_PATH)) as f:
        data = json.load(f)
    urls = sorted({item["website"] for item in data if "website" in item})
    print(f"Testing {len(urls)} unique URLs with max {MAX_CONCURRENT} concurrent sessions\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [test_single_url(url, sem) for url in urls]
    results: list[URLResult] = await asyncio.gather(*tasks)

    # Summary
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(ok)}/{len(results)} succeeded, {len(fail)} failed\n")

    # Captcha summary
    captcha_urls = [r for r in results if r.captcha_triggered]
    captcha_resolved = [r for r in captcha_urls if r.captcha_resolved]
    captcha_stuck = [r for r in captcha_urls if not r.captcha_resolved]
    if captcha_urls:
        print(f"CAPTCHA: {len(captcha_urls)} sites triggered, "
              f"{len(captcha_resolved)} resolved, {len(captcha_stuck)} stuck\n")
        for r in captcha_urls:
            status = "resolved" if r.captcha_resolved else "STUCK"
            print(f"  [{status}] {r.url}  (started={r.captcha_started_count} finished={r.captcha_finished_count})")
        print()
    else:
        print("CAPTCHA: No captcha events detected\n")

    if fail:
        # Group by error type
        by_type: dict[str, list[URLResult]] = {}
        for r in fail:
            by_type.setdefault(r.error_type, []).append(r)
        for etype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"  {etype} ({len(items)}):")
            for r in items:
                print(f"    - {r.url}")
                print(f"      {r.error[:200]}")
            print()

    # Write JSON results
    out_path = os.path.join(os.path.dirname(__file__), "..", "browserbase_url_test_results.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(
            [
                {
                    "url": r.url,
                    "success": r.success,
                    "session_id": r.session_id,
                    "error": r.error,
                    "error_type": r.error_type,
                    "page_title": r.page_title,
                    "elapsed_s": round(r.elapsed_s, 2),
                    "captcha_triggered": r.captcha_triggered,
                    "captcha_resolved": r.captcha_resolved,
                    "captcha_started_count": r.captcha_started_count,
                    "captcha_finished_count": r.captcha_finished_count,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
