from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import textwrap
import time
import traceback
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
    browserbase_enabled: bool = False
    browserbase_api_key: str = ""
    browserbase_project_id: str = ""
    browserbase_api_url: str = "https://api.browserbase.com/v1/sessions"
    browserbase_region: str | None = None
    browserbase_proxies: bool = False
    browserbase_keep_alive: bool = False
    browserbase_timeout_seconds: int = 1800


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
        self._step_index = 0
        self._prepared_task: dict[str, Any] = {}
        self._browserbase_session: dict[str, Any] | None = None

    def _runtime_log_path(self) -> Path:
        return self.config.output_dir / "runtime_errors.jsonl"

    def _log_browserbase_error(self, *, event: str, error: BaseException | str, **extra: Any) -> None:
        session_id = (self._browserbase_session or {}).get("id", "")
        append_runtime_log(
            self._runtime_log_path(),
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

    def prepare(self, **kwargs) -> None:
        self._prepared_task = dict(kwargs)
        start_url = kwargs.get("start_url") or self.config.start_url
        if start_url:
            self.config.start_url = start_url
        self._run(self._prepare_async(start_url=self.config.start_url))

    async def _prepare_async(self, start_url: str | None = None) -> None:
        if self._playwright is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            if self.config.browserbase_enabled:
                try:
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
                    if self._context.pages:
                        self._page = self._context.pages[0]
                    else:
                        self._page = await self._context.new_page()
                except Exception as exc:
                    self._log_browserbase_error(event="session_prepare_failed", error=exc)
                    raise
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
                self._page = await self._context.new_page()
            self._page.set_default_timeout(self.config.browser_timeout_ms)
            self._page.set_default_navigation_timeout(self.config.browser_navigation_timeout_ms)
            self._page.on("console", self._on_console)
            self._page.on("pageerror", self._on_page_error)

        if start_url:
            await self._page.goto(start_url, wait_until="domcontentloaded")
            await self._wait_for_rendered_page(start_url)

    async def _create_browserbase_session(self) -> dict[str, Any]:
        if not self.config.browserbase_api_key:
            raise RuntimeError("Missing BROWSERBASE_API_KEY for Browserbase session.")
        if not self.config.browserbase_project_id:
            raise RuntimeError("Missing BROWSERBASE_PROJECT_ID for Browserbase session.")

        payload: dict[str, Any] = {
            "projectId": self.config.browserbase_project_id,
            "keepAlive": self.config.browserbase_keep_alive,
            "timeout": self.config.browserbase_timeout_seconds,
            "proxies": self.config.browserbase_proxies,
        }
        if self.config.browserbase_region:
            payload["region"] = self.config.browserbase_region

        async with httpx.AsyncClient(timeout=self.config.browser_navigation_timeout_ms / 1000) as client:
            try:
                response = await client.post(
                    self.config.browserbase_api_url,
                    headers={"x-bb-api-key": self.config.browserbase_api_key},
                    json=payload,
                )
                response.raise_for_status()
                session = response.json()
            except Exception as exc:
                self._log_browserbase_error(event="session_create_failed", error=exc)
                raise

        if not session.get("connectUrl"):
            error = RuntimeError("Browserbase session response did not include connectUrl.")
            self._log_browserbase_error(event="session_create_failed", error=error)
            raise error
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
        async with httpx.AsyncClient(timeout=self.config.browser_navigation_timeout_ms / 1000) as client:
            try:
                response = await client.post(
                    release_url,
                    headers={"x-bb-api-key": self.config.browserbase_api_key},
                    json=payload,
                )
                response.raise_for_status()
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

    def _on_console(self, message) -> None:
        line = f"[{message.type}] {message.text}"
        self._console_history.append(line)
        self._step_console.append(line)

    def _on_page_error(self, error) -> None:
        line = f"[pageerror] {error}"
        self._console_history.append(line)
        self._step_console.append(line)

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        return self._run(self._execute_async(action))

    async def _execute_async(self, action: dict[str, Any]) -> dict[str, Any]:
        await self._prepare_async()
        self._step_index += 1
        self._step_console = []
        python_code = action.get("python_code", "")
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
        globals_dict: dict[str, Any] = {
            "asyncio": asyncio,
            "json": json,
            "re": re,
        }
        locals_dict: dict[str, Any] = {}
        wrapped = "async def __agent_step__(page, context, browser, playwright, task):\n"
        wrapped += textwrap.indent(python_code, "    ")
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

    async def _capture_observation(self, *, success: bool, exception_text: str) -> dict[str, Any]:
        screenshot_path: Path | None = self.config.output_dir / "screenshots" / f"step_{self._step_index:04d}.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.wait_for(
                self._page.screenshot(path=str(screenshot_path), full_page=False),
                timeout=self.config.observation_timeout_ms / 1000,
            )
        except Exception:
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
        self._run(self._close_async())
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    async def _close_async(self) -> None:
        try:
            if self._page is not None:
                await self._page.close()
            if self._context is not None:
                await self._context.close()
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception as exc:
            if self.config.browserbase_enabled:
                self._log_browserbase_error(event="session_close_failed", error=exc)
            raise
        finally:
            await self._release_browserbase_session()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
