from __future__ import annotations

import asyncio
import io
import inspect
import json
import os
import re
import textwrap
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from miniswewebagent.utils.logging import append_runtime_log


class LocalBrowserEnvironmentConfig(BaseModel):
    start_url: str | None = None
    headless: bool = False
    devtools: bool = False
    keep_open_on_exit: bool = False
    prompt_before_close: bool = False
    slow_mo_ms: int = 50
    browser_width: int = 1280
    browser_height: int = 1440
    browser_timeout_ms: int = 10000
    browser_navigation_timeout_ms: int = 30000
    step_execution_timeout_ms: int = 20000
    observation_timeout_ms: int = 5000
    output_dir: Path = Path("outputs/default")
    browserbase_enabled: bool = True
    browserbase_api_key: str = ""
    browserbase_project_id: str = ""
    browserbase_api_url: str = "https://api.browserbase.com/v1/sessions"
    browserbase_session_create_retries: int = 4
    browserbase_retry_backoff_seconds: float = 1.0


class LocalBrowserEnvironment:
    def __init__(self, *, config_class: type = LocalBrowserEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        if not self.config.browserbase_api_key:
            self.config.browserbase_api_key = os.environ.get("BROWSERBASE_API_KEY", "")
        if not self.config.browserbase_project_id:
            self.config.browserbase_project_id = os.environ.get("BROWSERBASE_PROJECT_ID", "")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._console_history: list[str] = []
        self._step_console: list[str] = []
        self._step_python_code = ""
        self._step_python_output = ""
        self._step_index = 0
        self._prepared_task: dict[str, Any] = {}
        self._browserbase_session: dict[str, Any] | None = None
        self._captcha_event = asyncio.Event()
        self._captcha_event.set()

    def _is_target_closed_error(self, error: BaseException | str) -> bool:
        text = str(error)
        if isinstance(error, BaseException) and type(error).__name__ == "TargetClosedError":
            return True
        return "Target page, context or browser has been closed" in text

    def _current_page_url(self) -> str:
        page = self._page
        if page is None:
            return self.config.start_url or ""
        try:
            url = getattr(page, "url", "")
        except Exception:
            url = ""
        return url or self.config.start_url or ""

    def _page_is_usable(self) -> bool:
        if self._page is None:
            return False
        checker = getattr(self._page, "is_closed", None)
        if callable(checker):
            try:
                return not checker()
            except Exception:
                return False
        return True

    def _context_is_usable(self) -> bool:
        if self._context is None:
            return False
        if not self._browser_is_usable():
            return False
        try:
            getattr(self._context, "pages", None)
        except Exception:
            return False
        return True

    def _browser_is_usable(self) -> bool:
        if self._browser is None:
            return False
        checker = getattr(self._browser, "is_connected", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return True

    async def _apply_page_viewport(self, page: Any | None) -> None:
        if page is None:
            return
        set_viewport_size = getattr(page, "set_viewport_size", None)
        if not callable(set_viewport_size):
            return
        await set_viewport_size(
            {
                "width": self.config.browser_width,
                "height": self.config.browser_height,
            }
        )

    async def _dispose_runtime_handles(self, *, stop_playwright: bool, release_browserbase: bool) -> None:
        page = self._page
        context = self._context
        browser = self._browser
        playwright = self._playwright
        session = self._browserbase_session

        self._page = None
        self._context = None
        self._browser = None
        if stop_playwright:
            self._playwright = None
        self._browserbase_session = None

        for handle in (page, context, browser):
            if handle is None:
                continue
            close = getattr(handle, "close", None)
            if not callable(close):
                continue
            try:
                await close()
            except Exception:
                pass

        if release_browserbase and self.config.browserbase_enabled and session is not None:
            self._browserbase_session = session
            try:
                await self._release_browserbase_session()
            except Exception:
                pass
            finally:
                self._browserbase_session = None

        if stop_playwright and playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass

    async def _ensure_browser_runtime(self) -> None:
        if self._playwright is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()

        if self._browser_is_usable() and self._context_is_usable() and self._page_is_usable():
            await self._apply_page_viewport(self._page)
            self._page.set_default_timeout(self.config.browser_timeout_ms)
            self._page.set_default_navigation_timeout(self.config.browser_navigation_timeout_ms)
            return

        if self._browser_is_usable() and self._context_is_usable() and not self._page_is_usable():
            try:
                existing_pages = [page for page in self._context.pages if not getattr(page, "is_closed", lambda: False)()]
            except Exception:
                existing_pages = []
            if existing_pages:
                self._page = existing_pages[0]
            else:
                self._page = await self._context.new_page()
            await self._apply_page_viewport(self._page)
            self._page.set_default_timeout(self.config.browser_timeout_ms)
            self._page.set_default_navigation_timeout(self.config.browser_navigation_timeout_ms)
            self._page.on("console", self._on_console)
            self._page.on("pageerror", self._on_page_error)
            return

        await self._dispose_runtime_handles(stop_playwright=False, release_browserbase=False)

        if self.config.browserbase_enabled:
            session = await self._create_browserbase_session()
            self._browserbase_session = session
            self._browser = await self._playwright.chromium.connect_over_cdp(
                session["connectUrl"],
                timeout=self.config.browser_navigation_timeout_ms,
            )
            if self._browser.contexts:
                self._context = self._browser.contexts[0]
            else:
                self._context = await self._browser.new_context(
                    viewport={
                        "width": self.config.browser_width,
                        "height": self.config.browser_height,
                    }
                )
        else:
            launch_args: list[str] = []
            if self.config.devtools:
                launch_args.append("--auto-open-devtools-for-tabs")

            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                args=launch_args,
                slow_mo=self.config.slow_mo_ms,
            )
            self._context = await self._browser.new_context(
                viewport={"width": self.config.browser_width, "height": self.config.browser_height}
            )

        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()
        await self._apply_page_viewport(self._page)
        self._page.set_default_timeout(self.config.browser_timeout_ms)
        self._page.set_default_navigation_timeout(self.config.browser_navigation_timeout_ms)
        self._page.on("console", self._on_console)
        self._page.on("pageerror", self._on_page_error)

    async def _recover_runtime(self, *, url_hint: str | None, source: str) -> bool:
        target_url = url_hint or self.config.start_url or ""
        self._log_runtime_event(source="browser", event="runtime_recovery_started", trigger=source, url=target_url)
        try:
            await self._dispose_runtime_handles(stop_playwright=True, release_browserbase=True)
            await self._ensure_browser_runtime()
            if target_url:
                await self._page.goto(target_url, wait_until="domcontentloaded")
                await self._wait_for_observation_ready()
        except Exception as exc:
            self._log_runtime_event(
                source="browser",
                event="runtime_recovery_failed",
                trigger=source,
                url=target_url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return False

        self._log_runtime_event(source="browser", event="runtime_recovery_succeeded", trigger=source, url=target_url)
        return True

    def _runtime_log_path(self) -> Path:
        return self.config.output_dir / "runtime_errors.jsonl"

    def _log_runtime_event(self, *, source: str, event: str, **extra: Any) -> None:
        append_runtime_log(
            self._runtime_log_path(),
            source=source,
            event=event,
            **extra,
        )

    def _log_browserbase_error(self, *, event: str, error: BaseException | str, **extra: Any) -> None:
        session_id = (self._browserbase_session or {}).get("id", "")
        self._log_runtime_event(
            source="browserbase",
            event=event,
            session_id=session_id,
            error_type=type(error).__name__ if isinstance(error, BaseException) else "Error",
            error=str(error),
            **extra,
        )

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro):
        loop = self._ensure_loop()
        return loop.run_until_complete(coro)

    def _is_retryable_browserbase_error(self, error: Exception) -> bool:
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            return status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        return False

    def _browserbase_retry_delay_seconds(self, attempt: int) -> float:
        base_delay = max(self.config.browserbase_retry_backoff_seconds, 0.0)
        if base_delay == 0:
            return 0.0
        return min(base_delay * (2 ** (attempt - 1)), 8.0)

    def prepare(self, **kwargs) -> None:
        self._prepared_task = dict(kwargs)
        start_url = kwargs.get("start_url") or self.config.start_url
        if start_url:
            self.config.start_url = start_url
        self._run(self._prepare_async(start_url=self.config.start_url))

    async def _prepare_async(self, start_url: str | None = None) -> None:
        try:
            await self._ensure_browser_runtime()
        except Exception as exc:
            if self.config.browserbase_enabled:
                self._log_browserbase_error(event="session_prepare_failed", error=exc)
            raise

        await self.wait_for_captcha_resolution()

        if start_url:
            await self._page.goto(start_url, wait_until="domcontentloaded")
            await self._wait_for_rendered_page(start_url)

    async def _create_browserbase_session(self) -> dict[str, Any]:
        if not self.config.browserbase_api_key:
            raise RuntimeError("Missing BROWSERBASE_API_KEY for Browserbase session.")
        if not self.config.browserbase_project_id:
            raise RuntimeError("Missing BROWSERBASE_PROJECT_ID for Browserbase session.")

        try:
            from browserbase import Browserbase
        except ImportError as exc:
            raise RuntimeError(
                "The browserbase package is required. Install dependencies with `pip install -e .`."
            ) from exc

        browserbase = Browserbase(api_key=self.config.browserbase_api_key)

        retry_attempts = max(self.config.browserbase_session_create_retries, 1)
        for attempt in range(1, retry_attempts + 1):
            try:
                created_session = await asyncio.to_thread(
                    browserbase.sessions.create,
                    project_id=self.config.browserbase_project_id,
                    proxies=True,
                    browser_settings={"advanced_stealth": True},
                    keep_alive=True,
                    timeout=1200,
                    region="us-east-1",
                )
                session = {
                    "id": getattr(created_session, "id", "") or created_session.get("id", ""),
                    "connectUrl": getattr(created_session, "connect_url", "") or created_session.get("connectUrl", ""),
                }
                break
            except Exception as exc:
                should_retry = attempt < retry_attempts and self._is_retryable_browserbase_error(exc)
                self._log_browserbase_error(
                    event="session_create_failed",
                    error=exc,
                    attempt=attempt,
                    retrying=should_retry,
                )
                if not should_retry:
                    raise
                await asyncio.sleep(self._browserbase_retry_delay_seconds(attempt))

        if not session.get("connectUrl"):
            error = RuntimeError("Browserbase session response did not include connectUrl.")
            self._log_browserbase_error(event="session_create_failed", error=error)
            raise error
        self._log_runtime_event(
            source="browserbase",
            event="session_created",
            session_id=session.get("id", ""),
            keep_alive=True,
            proxies=True,
            advanced_stealth=True,
            timeout=1200,
            region="us-east-1",
        )
        return session

    async def _release_browserbase_session(self) -> None:
        if not self.config.browserbase_enabled or not self._browserbase_session:
            return
        if not self.config.browserbase_api_key:
            return
        session_id = self._browserbase_session.get("id")
        if not session_id:
            return

        payload = {
            "projectId": self.config.browserbase_project_id,
            "status": "REQUEST_RELEASE",
        }
        release_url = f"{self.config.browserbase_api_url.rstrip('/')}/{session_id}"
        async with httpx.AsyncClient(
            timeout=self.config.browser_navigation_timeout_ms / 1000,
            trust_env=False,
        ) as client:
            try:
                response = await client.post(
                    release_url,
                    headers={"x-bb-api-key": self.config.browserbase_api_key},
                    json=payload,
                )
                response.raise_for_status()
                self._log_runtime_event(
                    source="browserbase",
                    event="session_release_requested",
                    session_id=session_id,
                )
            except Exception as exc:
                self._log_browserbase_error(event="session_release_failed", error=exc)

    async def _wait_for_rendered_page(self, start_url: str) -> None:
        for load_state in ("domcontentloaded", "load", "networkidle"):
            try:
                await self._page.wait_for_load_state(load_state, timeout=15000)
            except Exception:
                continue
        for _ in range(20):
            try:
                ready = await self._page.evaluate(
                    """
                    () => {
                      const href = window.location.href || "";
                      const title = document.title || "";
                      const body = document.body;
                      const text = body?.innerText?.trim() || "";
                      const count = body?.querySelectorAll?.('*')?.length || 0;
                      return href !== 'about:blank' && (title.length > 0 || text.length > 0 || count > 20);
                    }
                    """
                )
            except Exception:
                ready = False
            if ready:
                return
            await asyncio.sleep(0.5)
        raise RuntimeError(f"Page did not finish rendering for {start_url!r}.")

    async def _wait_for_observation_ready(self) -> None:
        timeout_ms = max(self.config.observation_timeout_ms, 1000)
        deadline = time.monotonic() + (timeout_ms / 1000)

        remaining_ms = max(int((deadline - time.monotonic()) * 1000), 1)
        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=min(remaining_ms, 3000))
        except Exception:
            return

        for load_state in ("load", "networkidle"):
            remaining_ms = max(int((deadline - time.monotonic()) * 1000), 1)
            if remaining_ms <= 1:
                break
            try:
                await self._page.wait_for_load_state(load_state, timeout=min(remaining_ms, 3000))
            except Exception:
                continue

        stable_samples = 0
        last_snapshot: tuple[str, str, str, int, int] | None = None
        settle_started = time.monotonic()
        minimum_settle_seconds = min(0.75, max((timeout_ms / 1000) * 0.5, 0.25))
        while time.monotonic() < deadline:
            remaining_seconds = max(deadline - time.monotonic(), 0.1)
            try:
                snapshot = await asyncio.wait_for(
                    self._page.evaluate(
                        """
                        () => {
                          const body = document.body;
                          const text = body?.innerText?.trim() || "";
                          return [
                            window.location.href || "",
                            document.readyState || "",
                            document.title || "",
                            body?.querySelectorAll?.('*')?.length || 0,
                            text.length,
                          ];
                        }
                        """
                    ),
                    timeout=min(remaining_seconds, 1.0),
                )
            except Exception:
                return

            current_snapshot = tuple(snapshot)
            ready_state = current_snapshot[1]
            text_length = current_snapshot[4]
            dom_count = current_snapshot[3]
            looks_ready = ready_state == "complete" and (text_length > 0 or dom_count > 20)
            settle_elapsed = time.monotonic() - settle_started

            if looks_ready and current_snapshot == last_snapshot and settle_elapsed >= minimum_settle_seconds:
                stable_samples += 1
                if stable_samples >= 2:
                    return
            else:
                stable_samples = 0

            last_snapshot = current_snapshot
            await asyncio.sleep(0.25)

    async def wait_for_captcha_resolution(self) -> None:
        await self._captcha_event.wait()

    def _on_console(self, message) -> None:
        line = f"[{message.type}] {message.text}"
        self._console_history.append(line)
        self._step_console.append(line)

        if message.text == "browserbase-solving-started":
            self._log_runtime_event(source="browserbase", event="captcha_solving_started")
            self._captcha_event.clear()
        elif message.text == "browserbase-solving-finished":
            self._log_runtime_event(source="browserbase", event="captcha_solving_finished")

            async def delayed_resume() -> None:
                await asyncio.sleep(3)
                try:
                    await self._page.wait_for_load_state("networkidle")
                except Exception:
                    pass
                self._captcha_event.set()

            asyncio.create_task(delayed_resume())

    def _on_page_error(self, error) -> None:
        line = f"[pageerror] {error}"
        self._console_history.append(line)
        self._step_console.append(line)

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        return self._run(self._execute_async(action))

    async def _execute_async(self, action: dict[str, Any]) -> dict[str, Any]:
        await self._prepare_async()
        await self.wait_for_captcha_resolution()
        self._step_index += 1
        self._step_console = []
        self._step_python_output = ""
        python_code = action.get("python_code", "")
        self._step_python_code = python_code
        self._persist_step_code(python_code)

        success = True
        exception_text = ""
        try:
            if python_code.strip():
                await asyncio.wait_for(
                    self._run_python_code(python_code),
                    timeout=self.config.step_execution_timeout_ms / 1000,
                )
            await self._wait_for_observation_ready()
        except Exception as exc:
            success = False
            exception_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ).strip()

        if exception_text and self._is_target_closed_error(exception_text):
            recovered = await self._recover_runtime(url_hint=self._current_page_url(), source="execute")
            if recovered:
                success = False
                exception_text = f"{exception_text}\n\n[recovery] Recovered browser runtime after closed target to capture observation."

        observation = await self._capture_observation(success=success, exception_text=exception_text)
        return {
            "output": json.dumps(observation, indent=2),
            "returncode": 0 if success else 1,
            "exception_info": exception_text,
            "observation": observation,
        }

    def _persist_step_code(self, python_code: str) -> None:
        steps_dir = self.config.output_dir / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        step_path = steps_dir / f"step_{self._step_index:04d}.py"
        step_path.write_text(python_code)

        cumulative_path = self.config.output_dir / "script.py"
        prefix = f"\n\n# Step {self._step_index}\n"
        with cumulative_path.open("a", encoding="utf-8") as handle:
            handle.write(prefix)
            handle.write(python_code)
            handle.write("\n")

    async def _run_python_code(self, python_code: str) -> None:
        buf = io.StringIO()
        globals_dict: dict[str, Any] = {
            "asyncio": asyncio,
            "json": json,
            "re": re,
        }
        locals_dict: dict[str, Any] = {}
        wrapped = "async def __agent_step__(page, context, browser, playwright, task):\n"
        wrapped += textwrap.indent(python_code, "    ")
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                exec(wrapped, globals_dict, locals_dict)
                result = locals_dict["__agent_step__"](
                    self._page,
                    self._context,
                    self._browser,
                    self._playwright,
                    self._prepared_task,
                )
                if inspect.isawaitable(result):
                    await result
        finally:
            self._step_python_output = buf.getvalue().strip()

    async def _capture_observation(self, *, success: bool, exception_text: str) -> dict[str, Any]:
        screenshot_path: Path | None = self.config.output_dir / "screenshots" / f"step_{self._step_index:04d}.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        capture_page = self._page

        async def attempt_screenshot() -> Path | None:
            if capture_page is None:
                return None
            await capture_page.locator("body").wait_for(state="attached", timeout=self.config.observation_timeout_ms)
            await asyncio.wait_for(
                capture_page.screenshot(path=str(screenshot_path), full_page=False),
                timeout=self.config.observation_timeout_ms / 1000,
            )
            return screenshot_path

        try:
            screenshot_path = await attempt_screenshot()
        except Exception as exc:
            if self._is_target_closed_error(exc):
                recovered = await self._recover_runtime(url_hint=self._current_page_url(), source="observation")
                capture_page = self._page
                if recovered:
                    try:
                        screenshot_path = await attempt_screenshot()
                    except Exception:
                        screenshot_path = None
                else:
                    screenshot_path = None
            else:
                screenshot_path = None

        try:
            aria_snapshot = await asyncio.wait_for(
                self._page.locator("body").aria_snapshot(),
                timeout=self.config.observation_timeout_ms / 1000,
            )
        except Exception as exc:
            aria_snapshot = f"<aria_snapshot_error>{exc}</aria_snapshot_error>"

        try:
            title = await asyncio.wait_for(
                self._page.title(),
                timeout=self.config.observation_timeout_ms / 1000,
            )
        except Exception as exc:
            if not exception_text:
                exception_text = f"Observation title capture failed: {exc}"
            title = ""

        return {
            "success": success,
            "exception": exception_text,
            "url": getattr(self._page, "url", ""),
            "title": title,
            "screenshot_path": str(screenshot_path) if screenshot_path is not None else "",
            "aria_snapshot": aria_snapshot,
            "python_code": self._step_python_code,
            "python_output": self._step_python_output,
            "console_output": "\n".join(self._step_console[-20:]),
            "recent_console": "\n".join(self._console_history[-50:]),
        }

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "start_url": self.config.start_url or "",
            "output_dir": str(self.config.output_dir),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "environment": {
                "config": self.config.model_dump(mode="json"),
                "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "browserbase_session": {
                    "id": (self._browserbase_session or {}).get("id", ""),
                },
            }
        }

    def close(self) -> None:
        if self.config.keep_open_on_exit:
            if self.config.prompt_before_close:
                prompt = "Browser kept open for debugging. Press Enter to close it..."
                try:
                    input(prompt)
                except EOFError:
                    return
            else:
                return
        try:
            self._run(self._close_async())
        finally:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.close()
            self._loop = None

    async def _close_async(self) -> None:
        close_error: Exception | None = None
        try:
            await self._dispose_runtime_handles(stop_playwright=True, release_browserbase=True)
        except Exception as exc:
            close_error = exc
            if self.config.browserbase_enabled:
                self._log_browserbase_error(event="session_close_failed", error=exc)
        finally:
            if self.config.browserbase_enabled:
                self._log_runtime_event(
                    source="browserbase",
                    event="session_close_complete",
                    session_id="",
                )
        if close_error is not None:
            raise close_error
