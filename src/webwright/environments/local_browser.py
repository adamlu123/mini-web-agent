from __future__ import annotations

import asyncio
import io
import json
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

_BROWSER_MODES = {"local_launch", "local_persistent"}


class LocalBrowserEnvironmentConfig(BaseModel):
    start_url: str | None = None
    browser_mode: str = "local_launch"
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
    user_data_dir: Path = Path("~/.cache/webwright/chrome-profile")
    launch_args: list[str] = Field(default_factory=list)

    @field_validator("browser_mode")
    @classmethod
    def validate_browser_mode(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if normalized not in _BROWSER_MODES:
            raise ValueError(
                f"browser_mode must be one of: {', '.join(sorted(_BROWSER_MODES))}"
            )
        return normalized


class LocalBrowserEnvironment:
    """Live local Playwright browser environment.

    The environment owns the browser/page and executes each model action as an async
    Python snippet with ``page``, ``context``, ``browser``, ``playwright``, and
    ``task`` already available.
    """

    def __init__(self, *, config_class: type = LocalBrowserEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.config.output_dir = self.config.output_dir.expanduser()
        self.config.user_data_dir = self.config.user_data_dir.expanduser()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._step_index = 0
        self._prepared_task: dict[str, Any] = {}
        self._console_history: list[str] = []
        self._step_console: list[str] = []
        self._step_python_code = ""
        self._step_python_output = ""

    def _screenshots_dir(self) -> Path:
        return self.config.output_dir / "screenshots"

    def _steps_dir(self) -> Path:
        return self.config.output_dir / "steps"

    def prepare(self, **kwargs) -> None:
        self._prepared_task = dict(kwargs)
        self._step_index = 0
        self._console_history = []
        self._step_console = []
        start_url = kwargs.get("start_url") or self.config.start_url
        if start_url:
            self.config.start_url = str(start_url)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._steps_dir().mkdir(parents=True, exist_ok=True)
        self._screenshots_dir().mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "task.json").write_text(
            json.dumps(kwargs, indent=2),
            encoding="utf-8",
        )
        self._run(self._prepare_async())

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro):
        loop = self._ensure_loop()
        return loop.run_until_complete(coro)

    async def _prepare_async(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        chromium = self._playwright.chromium
        launch_args = list(self.config.launch_args)
        if self.config.devtools:
            launch_args.append("--auto-open-devtools-for-tabs")
        launch_kwargs = {
            "headless": self.config.headless,
            "slow_mo": self.config.slow_mo_ms,
            "args": launch_args,
        }

        if self.config.browser_mode == "local_persistent":
            self.config.user_data_dir.mkdir(parents=True, exist_ok=True)
            self._context = await chromium.launch_persistent_context(
                user_data_dir=str(self.config.user_data_dir),
                viewport={
                    "width": self.config.browser_width,
                    "height": self.config.browser_height,
                },
                **launch_kwargs,
            )
            self._browser = self._context.browser
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            self._browser = await chromium.launch(**launch_kwargs)
            self._context = await self._browser.new_context(
                viewport={
                    "width": self.config.browser_width,
                    "height": self.config.browser_height,
                }
            )
            self._page = await self._context.new_page()

        self._context.set_default_timeout(self.config.browser_timeout_ms)
        self._context.set_default_navigation_timeout(self.config.browser_navigation_timeout_ms)
        self._attach_page_listeners(self._page)
        if self.config.start_url:
            await self._page.goto(self.config.start_url, wait_until="domcontentloaded")

    def _attach_page_listeners(self, page: Any) -> None:
        page.on("console", self._on_console_message)
        page.on("pageerror", self._on_page_error)

    def _on_console_message(self, message: Any) -> None:
        text = getattr(message, "text", "")
        if callable(text):
            text = text()
        line = str(text)
        self._console_history.append(line)
        self._step_console.append(line)

    def _on_page_error(self, error: Any) -> None:
        line = f"Page error: {error}"
        self._console_history.append(line)
        self._step_console.append(line)

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        del cwd
        return self._run(self._execute_async(action))

    async def _execute_async(self, action: dict[str, Any]) -> dict[str, Any]:
        self._step_index += 1
        self._step_console = []
        self._step_python_output = ""
        self._step_python_code = str(action.get("python_code", "") or "")
        self._persist_step_code(self._step_python_code)

        success = True
        exception_text = ""
        try:
            if self._step_python_code.strip():
                await asyncio.wait_for(
                    self._run_python_code(self._step_python_code),
                    timeout=self.config.step_execution_timeout_ms / 1000,
                )
            await self._wait_for_observation_ready()
        except Exception:
            success = False
            exception_text = traceback.format_exc()

        observation = await self._capture_observation(
            success=success,
            exception_text=exception_text,
        )
        return {
            "output": self._step_python_output,
            "returncode": 0 if success else 1,
            "exception_info": exception_text,
            "observation": observation,
        }

    def _persist_step_code(self, python_code: str) -> None:
        step_path = self._steps_dir() / f"step_{self._step_index:04d}.py"
        step_path.write_text(python_code, encoding="utf-8")

        script_path = self.config.output_dir / "script.py"
        with script_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n# Step {self._step_index}\n")
            handle.write(python_code)
            handle.write("\n")

    async def _run_python_code(self, python_code: str) -> None:
        if self._page is None or self._context is None or self._playwright is None:
            raise RuntimeError("Browser environment was not prepared.")

        buffer = io.StringIO()
        globals_dict: dict[str, Any] = {"asyncio": asyncio}
        locals_dict: dict[str, Any] = {}
        wrapped = "async def __agent_step__(page, context, browser, playwright, task):\n"
        wrapped += textwrap.indent(python_code, "    ")
        with redirect_stdout(buffer), redirect_stderr(buffer):
            exec(wrapped, globals_dict, locals_dict)
            await locals_dict["__agent_step__"](
                self._page,
                self._context,
                self._browser,
                self._playwright,
                self._prepared_task,
            )
        self._step_python_output = buffer.getvalue()

    async def _wait_for_observation_ready(self) -> None:
        if self._page is None:
            return
        try:
            await self._page.wait_for_load_state(
                "domcontentloaded",
                timeout=self.config.observation_timeout_ms,
            )
        except Exception:
            pass

    async def _capture_observation(self, *, success: bool, exception_text: str) -> dict[str, Any]:
        page = self._page
        url = ""
        title = ""
        aria_snapshot = ""
        screenshot_path: Path | None = None

        if page is not None:
            try:
                url = page.url
            except Exception:
                url = self.config.start_url or ""
            try:
                title = await page.title()
            except Exception:
                title = ""
            try:
                aria_snapshot = await page.locator("body").aria_snapshot(
                    timeout=self.config.observation_timeout_ms,
                )
            except Exception:
                aria_snapshot = ""
            try:
                screenshot_path = self._screenshots_dir() / f"step_{self._step_index:04d}.png"
                await page.screenshot(path=str(screenshot_path), full_page=False)
            except Exception:
                screenshot_path = None

        return {
            "success": success,
            "exception": exception_text,
            "url": url or self.config.start_url or "",
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
            "output_dir": str(self.config.output_dir.resolve()),
            "browser_mode": self.config.browser_mode,
            "user_data_dir": str(self.config.user_data_dir),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "environment": {
                "config": self.config.model_dump(mode="json"),
                "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            }
        }

    def close(self) -> None:
        if self.config.prompt_before_close:
            input("Press Enter to close the browser...")
        if self.config.keep_open_on_exit:
            return
        try:
            self._run(self._close_async())
        finally:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.close()
            self._loop = None

    async def _close_async(self) -> None:

        context = self._context
        browser = self._browser
        playwright = self._playwright
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

        try:
            if context is not None:
                await context.close()
            elif browser is not None:
                await browser.close()
        finally:
            if playwright is not None:
                await playwright.stop()
