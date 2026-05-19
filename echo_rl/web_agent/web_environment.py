"""WebAgentEnvironment — terminal-style facade over a single Playwright tab.

This module gives the agent the same ``setup() / exec(cmd) / run_verifier() /
cleanup()`` shape as ``HarborEnvironment`` so the rollout loop can be reused
verbatim. ``exec`` interprets a shell command in one of three ways:

1. ``web <subcommand> [args]`` — high-level browser actions implemented inline.
2. ``python -c '<code>'`` or ``python <file>`` — run Python with ``page``,
   ``context``, ``browser`` and ``task`` already in scope. This is the
   "execute some python script/command to drive browser action" path.
3. Anything else — ``subprocess`` run inside the rollout workspace.

Output is rendered as plain text (URL, title, ARIA snapshot, console excerpt)
so it can be appended to the agent transcript like a real terminal output.

For unit tests, ``WebAgentEnvironment(stub=True)`` swaps the Playwright tab
for an in-memory fake that records calls and returns canned snapshots — this
keeps the smoke test runnable without internet / a real browser install.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass
class _StepRecord:
    turn: int
    command: str
    action_summary: str
    screenshot_path: str
    url: str
    title: str
    exception: str = ""


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[:max_chars]
    return head + f"\n[Output truncated: showing first {max_chars} of {len(text)} chars]"


def _unquote(value: str) -> str:
    """Strip a matching pair of outer single or double quotes from a CLI arg."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


class _StubPage:
    """Minimal Playwright-page lookalike used for tests."""

    def __init__(self, start_url: str) -> None:
        self.url = start_url or "about:blank"
        self._title = "Stub Page"
        self.actions: list[tuple[str, dict[str, Any]]] = []

    async def goto(self, url: str, **kwargs: Any) -> None:
        self.actions.append(("goto", {"url": url}))
        self.url = url

    async def title(self) -> str:
        return self._title

    async def click(self, selector: str, **kwargs: Any) -> None:
        self.actions.append(("click", {"selector": selector}))

    async def fill(self, selector: str, value: str, **kwargs: Any) -> None:
        self.actions.append(("fill", {"selector": selector, "value": value}))

    async def keyboard_press(self, key: str) -> None:
        self.actions.append(("press", {"key": key}))

    async def wait_for_selector(self, selector: str, **kwargs: Any) -> None:
        self.actions.append(("wait", {"selector": selector}))

    async def go_back(self) -> None:
        self.actions.append(("back", {}))

    async def go_forward(self) -> None:
        self.actions.append(("forward", {}))

    async def reload(self) -> None:
        self.actions.append(("reload", {}))

    async def screenshot(self, path: str, **kwargs: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # 1x1 PNG so judges that demand a real image can decode it.
        Path(path).write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
            b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
            b"\x84I\x85\x91\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    async def evaluate(self, expression: str) -> Any:
        self.actions.append(("evaluate", {"expression": expression}))
        return f"<stub:{expression[:80]}>"

    async def inner_text(self, selector: str) -> str:
        return f"<stub-text for {selector}>"

    def aria_snapshot(self) -> str:  # not async for the stub; matches locator().aria_snapshot()
        return f"- generic [stub aria snapshot for {self.url}]"


class _StubLocator:
    def __init__(self, page: _StubPage, selector: str) -> None:
        self._page = page
        self._selector = selector

    async def aria_snapshot(self) -> str:
        return self._page.aria_snapshot()

    async def wait_for(self, **kwargs: Any) -> None:
        return None


def _attach_stub_locator(page: _StubPage) -> None:
    def _locator(selector: str) -> _StubLocator:
        return _StubLocator(page, selector)

    page.locator = _locator  # type: ignore[attr-defined]


@dataclass
class WebAgentEnvironmentConfig:
    task_id: str = ""
    task: str = ""
    start_url: str = ""
    headless: bool = True
    browser_width: int = 1280
    browser_height: int = 1024
    nav_timeout_ms: int = 30000
    op_timeout_ms: int = 15000
    workspace_root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "web_agent_workspace")
    screenshots_dirname: str = "screenshots"
    max_snapshot_chars: int = 6000
    max_text_chars: int = 4000
    max_shell_chars: int = 6000
    use_browserbase: bool = False
    browserbase_api_key: str = ""
    browserbase_project_id: str = ""
    stub: bool = False


class WebAgentEnvironment:
    """Playwright-backed terminal sandbox."""

    def __init__(self, cfg: WebAgentEnvironmentConfig) -> None:
        self.cfg = cfg
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._is_setup = False
        self._turn = 0
        self._steps: list[_StepRecord] = []
        self._workspace: Path | None = None
        self._screenshots_dir: Path | None = None

    # ------------------------------------------------------------------
    # public API mirrors HarborEnvironment
    # ------------------------------------------------------------------
    async def setup(self) -> None:
        if self._is_setup:
            return
        suffix = (self.cfg.task_id or "task").replace("/", "_")[:40]
        self._workspace = self.cfg.workspace_root / f"{suffix}_{int(time.time() * 1000)}"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._screenshots_dir = self._workspace / self.cfg.screenshots_dirname
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.stub:
            self._page = _StubPage(self.cfg.start_url)
            _attach_stub_locator(self._page)
        else:
            await self._start_playwright()
            if self.cfg.start_url:
                await self._page.goto(self.cfg.start_url, wait_until="domcontentloaded")
        self._is_setup = True
        # Persist a record of starting state so we don't lose the entry frame.
        await self._capture_screenshot("start")

    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        if not self._is_setup:
            raise RuntimeError("setup() must be called before exec().")
        self._turn += 1
        timeout = float(timeout) if timeout is not None else 30.0
        command = command.strip()
        if not command:
            return ExecResult(stdout="", stderr="empty command", return_code=2)

        try:
            if command.startswith("web "):
                result = await asyncio.wait_for(self._handle_web(command[4:].strip()), timeout=timeout)
            elif command == "web":
                result = ExecResult(stdout=self._web_help(), return_code=0)
            elif command.startswith("python ") or command.startswith("python3 "):
                result = await asyncio.wait_for(self._handle_python_cli(command), timeout=timeout)
            else:
                result = await asyncio.wait_for(self._handle_shell(command, timeout=timeout), timeout=timeout)
        except asyncio.TimeoutError:
            result = ExecResult(
                stdout="",
                stderr=f"command timed out after {timeout}s",
                return_code=124,
            )

        # Always take a turn-end screenshot so the judge sees the new page.
        screenshot_path = await self._capture_screenshot(self._turn_label())
        url, title = await self._safe_url_title()
        self._steps.append(
            _StepRecord(
                turn=self._turn,
                command=command,
                action_summary=self._summarize_action(command),
                screenshot_path=str(screenshot_path) if screenshot_path else "",
                url=url,
                title=title,
                exception=result.stderr if result.return_code != 0 else "",
            )
        )
        if result.return_code == 0 and url:
            result.stdout += f"\n\nURL: {url}\nTitle: {title}"
        return result

    async def run_verifier(self, timeout: float | None = None) -> tuple[float, str | None]:
        # The web-agent has no in-env verifier; the reward function is called
        # outside this object by the rollout glue. Return a sentinel so the
        # generator knows to skip the in-env path.
        del timeout
        return 0.0, "no_in_env_verifier"

    async def cleanup(self) -> None:
        if not self.cfg.stub:
            await self._stop_playwright()
        self._is_setup = False
        # Workspace deliberately left on disk so a judge / human reviewer can
        # inspect screenshots and step logs afterwards.

    # ------------------------------------------------------------------
    # rollout-summary helpers (used by reward + smoke test)
    # ------------------------------------------------------------------
    @property
    def workspace(self) -> Path:
        if self._workspace is None:
            raise RuntimeError("Environment not set up yet.")
        return self._workspace

    @property
    def screenshots_dir(self) -> Path:
        if self._screenshots_dir is None:
            raise RuntimeError("Environment not set up yet.")
        return self._screenshots_dir

    def actions_history(self) -> list[str]:
        return [
            f"step {step.turn} action: {step.action_summary}"
            for step in self._steps
            if step.turn >= 1
        ]

    def screenshot_paths(self) -> list[str]:
        return [step.screenshot_path for step in self._steps if step.screenshot_path]

    # ------------------------------------------------------------------
    # web subcommands
    # ------------------------------------------------------------------
    def _web_help(self) -> str:
        return (
            "web subcommands: goto <url> | url | title | snapshot [max] | text [max] | "
            "click <sel> | fill <sel> <val> | press <key> | wait <sel> [ms] | "
            "back | forward | reload | screenshot [name] | eval <js> | py <code>"
        )

    async def _handle_web(self, args: str) -> ExecResult:
        if not args:
            return ExecResult(stdout=self._web_help(), return_code=0)
        try:
            sub, rest = self._split_subcommand(args)
        except ValueError as exc:
            return ExecResult(stderr=str(exc), return_code=2)

        try:
            handler = self._web_handlers().get(sub)
            if handler is None:
                return ExecResult(stderr=f"unknown web subcommand: {sub}\n{self._web_help()}", return_code=2)
            return await handler(rest)
        except Exception as exc:
            return ExecResult(stderr=f"web {sub} error: {exc}", return_code=1)

    def _web_handlers(self) -> dict[str, Any]:
        return {
            "goto": self._web_goto,
            "url": self._web_url,
            "title": self._web_title,
            "snapshot": self._web_snapshot,
            "text": self._web_text,
            "click": self._web_click,
            "fill": self._web_fill,
            "press": self._web_press,
            "wait": self._web_wait,
            "back": self._web_back,
            "forward": self._web_forward,
            "reload": self._web_reload,
            "screenshot": self._web_screenshot,
            "eval": self._web_eval,
            "py": self._web_py,
        }

    async def _web_goto(self, rest: str) -> ExecResult:
        if not rest:
            return ExecResult(stderr="usage: web goto <url>", return_code=2)
        url = rest.strip().strip("'\"")
        await self._page.goto(url, wait_until="domcontentloaded")
        return ExecResult(stdout=f"navigated to {url}", return_code=0)

    async def _web_url(self, rest: str) -> ExecResult:
        del rest
        url = getattr(self._page, "url", "")
        return ExecResult(stdout=str(url), return_code=0)

    async def _web_title(self, rest: str) -> ExecResult:
        del rest
        title_or_coro = self._page.title()
        title = await title_or_coro if asyncio.iscoroutine(title_or_coro) else title_or_coro
        return ExecResult(stdout=str(title), return_code=0)

    async def _web_snapshot(self, rest: str) -> ExecResult:
        max_chars = self.cfg.max_snapshot_chars
        if rest.strip().isdigit():
            max_chars = int(rest.strip())
        locator = self._page.locator("body") if hasattr(self._page, "locator") else None
        try:
            if locator is None:
                snapshot = self._page.aria_snapshot()  # stub path
            else:
                snapshot = await locator.aria_snapshot()
        except Exception as exc:
            snapshot = f"<aria_snapshot_error>{exc}</aria_snapshot_error>"
        return ExecResult(stdout=_truncate(str(snapshot), max_chars), return_code=0)

    async def _web_text(self, rest: str) -> ExecResult:
        max_chars = self.cfg.max_text_chars
        if rest.strip().isdigit():
            max_chars = int(rest.strip())
        try:
            text = await self._page.inner_text("body")
        except Exception as exc:
            return ExecResult(stderr=f"could not read body text: {exc}", return_code=1)
        return ExecResult(stdout=_truncate(text, max_chars), return_code=0)

    async def _web_click(self, rest: str) -> ExecResult:
        if not rest:
            return ExecResult(stderr="usage: web click <selector>", return_code=2)
        selector = _unquote(rest)
        await self._page.click(selector)
        return ExecResult(stdout=f"clicked {selector}", return_code=0)

    async def _web_fill(self, rest: str) -> ExecResult:
        if " " not in rest:
            return ExecResult(stderr="usage: web fill <selector> <value>", return_code=2)
        selector, value = rest.split(" ", 1)
        selector = _unquote(selector)
        value = _unquote(value)
        await self._page.fill(selector, value)
        return ExecResult(stdout=f"filled {selector!r} with {len(value)} chars", return_code=0)

    async def _web_press(self, rest: str) -> ExecResult:
        if not rest:
            return ExecResult(stderr="usage: web press <key>", return_code=2)
        if hasattr(self._page, "keyboard"):
            await self._page.keyboard.press(rest)
        else:  # stub path uses keyboard_press
            await self._page.keyboard_press(rest)
        return ExecResult(stdout=f"pressed {rest}", return_code=0)

    async def _web_wait(self, rest: str) -> ExecResult:
        if not rest:
            return ExecResult(stderr="usage: web wait <selector> [timeout_ms]", return_code=2)
        parts = rest.rsplit(" ", 1)
        timeout_ms = self.cfg.op_timeout_ms
        selector = rest
        if len(parts) == 2 and parts[1].isdigit():
            selector = parts[0]
            timeout_ms = int(parts[1])
        selector = _unquote(selector)
        await self._page.wait_for_selector(selector, timeout=timeout_ms)
        return ExecResult(stdout=f"selector visible: {selector}", return_code=0)

    async def _web_back(self, rest: str) -> ExecResult:
        del rest
        await self._page.go_back()
        return ExecResult(stdout="went back", return_code=0)

    async def _web_forward(self, rest: str) -> ExecResult:
        del rest
        await self._page.go_forward()
        return ExecResult(stdout="went forward", return_code=0)

    async def _web_reload(self, rest: str) -> ExecResult:
        del rest
        await self._page.reload()
        return ExecResult(stdout="reloaded", return_code=0)

    async def _web_screenshot(self, rest: str) -> ExecResult:
        name = rest.strip() or self._turn_label()
        path = await self._capture_screenshot(name, increment_turn=False)
        if path is None:
            return ExecResult(stderr="screenshot failed", return_code=1)
        return ExecResult(stdout=f"saved {path}", return_code=0)

    async def _web_eval(self, rest: str) -> ExecResult:
        if not rest:
            return ExecResult(stderr="usage: web eval <js_expression>", return_code=2)
        try:
            value = await self._page.evaluate(rest)
        except Exception as exc:
            return ExecResult(stderr=f"eval error: {exc}", return_code=1)
        return ExecResult(stdout=json.dumps(value, default=str)[: self.cfg.max_shell_chars], return_code=0)

    async def _web_py(self, rest: str) -> ExecResult:
        if not rest:
            return ExecResult(stderr="usage: web py <python_code>", return_code=2)
        return await self._run_python_snippet(rest)

    # ------------------------------------------------------------------
    # python / shell fallbacks
    # ------------------------------------------------------------------
    async def _handle_python_cli(self, command: str) -> ExecResult:
        # Accept `python -c '...'` and `python <file>`.
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return ExecResult(stderr=f"could not parse python command: {exc}", return_code=2)
        if len(argv) >= 3 and argv[1] == "-c":
            return await self._run_python_snippet(argv[2])
        if len(argv) >= 2 and argv[1].endswith(".py"):
            script_path = self._resolve_workspace_path(argv[1])
            if not script_path.exists():
                return ExecResult(stderr=f"script not found: {script_path}", return_code=2)
            return await self._run_python_snippet(script_path.read_text())
        # Anything else (e.g. `python -m foo`) falls back to subprocess.
        return await self._handle_shell(command)

    async def _run_python_snippet(self, code: str) -> ExecResult:
        import io
        from contextlib import redirect_stdout, redirect_stderr

        out_buf, err_buf = io.StringIO(), io.StringIO()
        globals_dict: dict[str, Any] = {
            "asyncio": asyncio,
            "json": json,
            "page": self._page,
            "context": self._context,
            "browser": self._browser,
            "task": self.cfg.task,
            "start_url": self.cfg.start_url,
        }
        locals_dict: dict[str, Any] = {}
        wrapper = "async def __web_step__():\n" + textwrap.indent(code, "    ")
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                exec(wrapper, globals_dict, locals_dict)
                coro = locals_dict["__web_step__"]()
                if asyncio.iscoroutine(coro):
                    await coro
        except Exception as exc:
            return ExecResult(
                stdout=out_buf.getvalue(),
                stderr=(err_buf.getvalue() + "\n" + str(exc)).strip(),
                return_code=1,
            )
        return ExecResult(stdout=out_buf.getvalue(), stderr=err_buf.getvalue(), return_code=0)

    async def _handle_shell(self, command: str, timeout: float = 30.0) -> ExecResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(stderr=f"shell command timed out after {timeout}s", return_code=124)
        return ExecResult(
            stdout=_truncate(stdout.decode(errors="replace"), self.cfg.max_shell_chars),
            stderr=_truncate(stderr.decode(errors="replace"), self.cfg.max_shell_chars),
            return_code=proc.returncode or 0,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _split_subcommand(args: str) -> tuple[str, str]:
        args = args.strip()
        if " " in args:
            sub, rest = args.split(" ", 1)
            return sub, rest.strip()
        return args, ""

    def _summarize_action(self, command: str) -> str:
        if command.startswith("web "):
            return command  # short and meaningful
        if command.startswith("python -c"):
            return "python -c '<inline code>'"
        # Truncate very long shell commands so they don't dwarf the judge prompt.
        return command if len(command) <= 200 else command[:197] + "..."

    def _turn_label(self) -> str:
        return f"final_execution_{self._turn:04d}"

    def _resolve_workspace_path(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = self.workspace / path
        return path

    async def _capture_screenshot(self, name: str, *, increment_turn: bool = True) -> Path | None:
        del increment_turn
        if self._screenshots_dir is None:
            return None
        out_path = self._screenshots_dir / f"{name}.png"
        try:
            await self._page.screenshot(path=str(out_path), full_page=False)
        except Exception as exc:
            logger.warning("screenshot failed for %s: %s", name, exc)
            return None
        return out_path

    async def _safe_url_title(self) -> tuple[str, str]:
        url = getattr(self._page, "url", "") or ""
        title = ""
        try:
            title_or_coro = self._page.title()
            title = await title_or_coro if asyncio.iscoroutine(title_or_coro) else title_or_coro
        except Exception:
            title = ""
        return url, str(title)

    async def _start_playwright(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "playwright is required for WebAgentEnvironment(stub=False). "
                "Install with `pip install playwright && playwright install chromium`."
            ) from exc

        self._playwright = await async_playwright().start()
        if self.cfg.use_browserbase:
            self._browser, self._context = await self._connect_browserbase()
        else:
            self._browser = await self._playwright.chromium.launch(headless=self.cfg.headless)
            self._context = await self._browser.new_context(
                viewport={"width": self.cfg.browser_width, "height": self.cfg.browser_height},
            )
        pages = list(self._context.pages)
        self._page = pages[0] if pages else await self._context.new_page()
        self._page.set_default_timeout(self.cfg.op_timeout_ms)
        self._page.set_default_navigation_timeout(self.cfg.nav_timeout_ms)

    async def _connect_browserbase(self):
        try:
            from browserbase import Browserbase
        except ImportError as exc:
            raise RuntimeError(
                "browserbase is required when use_browserbase=True. Install via `pip install browserbase`."
            ) from exc
        api_key = self.cfg.browserbase_api_key or ""
        project_id = self.cfg.browserbase_project_id or ""
        if not (api_key and project_id):
            raise RuntimeError(
                "use_browserbase=True requires browserbase_api_key and browserbase_project_id."
            )
        client = Browserbase(api_key=api_key)
        session = await asyncio.to_thread(
            client.sessions.create,
            project_id=project_id,
            proxies=True,
            browser_settings={"advanced_stealth": True},
            keep_alive=True,
            timeout=1200,
        )
        connect_url = getattr(session, "connect_url", "") or session.get("connectUrl", "")
        browser = await self._playwright.chromium.connect_over_cdp(
            connect_url, timeout=self.cfg.nav_timeout_ms
        )
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        return browser, context

    async def _stop_playwright(self) -> None:
        for handle in (self._page, self._context, self._browser):
            if handle is None:
                continue
            close = getattr(handle, "close", None)
            if callable(close):
                try:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass
        self._page = None
        self._context = None
        self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
