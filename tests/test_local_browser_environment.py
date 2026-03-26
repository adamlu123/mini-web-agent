from __future__ import annotations

import sys
import types
from pathlib import Path
from time import monotonic

import httpx

from miniswewebagent.environments.local_browser import LocalBrowserEnvironment


def test_local_browser_environment_runs_multiple_steps(tmp_path: Path) -> None:
    html_path = tmp_path / "page.html"
    html_path.write_text(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Local Browser Probe</title>
  </head>
  <body>
    <main>
      <h1 id="title">Counter</h1>
      <p id="value">0</p>
      <button id="increment">Increment</button>
    </main>
    <script>
      const button = document.getElementById('increment');
      const value = document.getElementById('value');
      button.addEventListener('click', () => {
        value.textContent = String(Number(value.textContent) + 1);
        console.log(`counter=${value.textContent}`);
      });
    </script>
  </body>
</html>
        """.strip(),
        encoding="utf-8",
    )

    env = LocalBrowserEnvironment(
        browserbase_enabled=False,
        headless=True,
        devtools=False,
        keep_open_on_exit=False,
        prompt_before_close=False,
        slow_mo_ms=0,
        browser_timeout_ms=3000,
        browser_navigation_timeout_ms=3000,
        step_execution_timeout_ms=3000,
        observation_timeout_ms=2000,
        output_dir=tmp_path / "artifacts",
    )

    env.prepare(start_url=html_path.resolve().as_uri(), task="probe", task_id="probe")

    step1 = env.execute(
        {
            "python_code": """
await page.get_by_role('button', name='Increment').click()
""".strip()
        }
    )
    assert step1["returncode"] == 0
    assert step1["observation"]["success"] is True
    assert "counter=1" in step1["observation"]["recent_console"]

    step2 = env.execute(
        {
            "python_code": """
value = await page.locator('#value').text_content()
assert value == '1', value
""".strip()
        }
    )
    assert step2["returncode"] == 0
    assert step2["observation"]["success"] is True

    step3 = env.execute(
        {
            "python_code": """
await page.get_by_role('button', name='Increment').click()
value = await page.locator('#value').text_content()
assert value == '2', value
""".strip()
        }
    )
    assert step3["returncode"] == 0
    assert step3["observation"]["success"] is True
    assert "counter=2" in step3["observation"]["recent_console"]

    env.close()

    assert (tmp_path / "artifacts" / "steps" / "step_0001.py").exists()
    assert (tmp_path / "artifacts" / "steps" / "step_0002.py").exists()
    assert (tmp_path / "artifacts" / "steps" / "step_0003.py").exists()


def test_local_browser_environment_waits_for_post_action_render(tmp_path: Path) -> None:
        html_path = tmp_path / "delayed.html"
        html_path.write_text(
                """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Delayed Render</title>
    </head>
    <body>
        <main>
            <p id="status">Idle</p>
            <button id="start">Start</button>
        </main>
        <script>
            document.getElementById('start').addEventListener('click', () => {
                setTimeout(() => {
                    document.title = 'Delayed Render Complete';
                    document.getElementById('status').textContent = 'Finished loading';
                    console.log('delayed-render-complete');
                }, 700);
            });
        </script>
    </body>
</html>
                """.strip(),
                encoding="utf-8",
        )

        env = LocalBrowserEnvironment(
            browserbase_enabled=False,
                headless=True,
                devtools=False,
                keep_open_on_exit=False,
                prompt_before_close=False,
                slow_mo_ms=0,
                browser_timeout_ms=3000,
                browser_navigation_timeout_ms=3000,
                step_execution_timeout_ms=3000,
                observation_timeout_ms=2000,
                output_dir=tmp_path / "artifacts",
        )

        env.prepare(start_url=html_path.resolve().as_uri(), task="probe", task_id="probe")
        step = env.execute(
                {
                        "python_code": """
await page.get_by_role('button', name='Start').click()
""".strip()
                }
        )

        assert step["returncode"] == 0
        assert step["observation"]["success"] is True
        assert step["observation"]["title"] == "Delayed Render Complete"
        assert "Finished loading" in step["observation"]["aria_snapshot"]
        assert "delayed-render-complete" in step["observation"]["recent_console"]

        env.close()


def test_observation_ready_does_not_spin_until_navigation_timeout(tmp_path: Path) -> None:
    class FakePage:
        def __init__(self) -> None:
            self.load_states: list[str] = []
            self.evaluate_calls = 0

        async def wait_for_load_state(self, load_state: str, timeout: int = 0) -> None:
            self.load_states.append(load_state)

        async def evaluate(self, script: str):
            self.evaluate_calls += 1
            return ["https://example.com", "complete", "Ready", 25, 100]

    env = LocalBrowserEnvironment(
        observation_timeout_ms=800,
        browser_navigation_timeout_ms=45000,
        output_dir=tmp_path / "artifacts",
    )
    env._page = FakePage()

    started = monotonic()
    env._run(env._wait_for_observation_ready())
    elapsed = monotonic() - started

    assert elapsed < 2
    assert env._page.load_states.count("domcontentloaded") == 1
    assert env._page.evaluate_calls >= 2


def test_local_browser_environment_prepare_navigates_on_reuse(tmp_path: Path) -> None:
    first_path = tmp_path / "first.html"
    first_path.write_text(
        "<html><head><title>First</title></head><body><h1>First</h1></body></html>",
        encoding="utf-8",
    )
    second_path = tmp_path / "second.html"
    second_path.write_text(
        "<html><head><title>Second</title></head><body><h1>Second</h1></body></html>",
        encoding="utf-8",
    )

    env = LocalBrowserEnvironment(
        browserbase_enabled=False,
        headless=True,
        devtools=False,
        keep_open_on_exit=False,
        prompt_before_close=False,
        slow_mo_ms=0,
        browser_timeout_ms=3000,
        browser_navigation_timeout_ms=3000,
        step_execution_timeout_ms=3000,
        observation_timeout_ms=2000,
        output_dir=tmp_path / "artifacts",
    )

    env.prepare(start_url=first_path.resolve().as_uri(), task="first", task_id="first")
    first_observation = env.execute({"python_code": ""})
    assert first_observation["observation"]["title"] == "First"

    env.prepare(start_url=second_path.resolve().as_uri(), task="second", task_id="second")
    second_observation = env.execute({"python_code": ""})
    assert second_observation["observation"]["title"] == "Second"

    env.close()


def test_local_browser_environment_uses_browserbase_cdp(monkeypatch, tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []

    class FakePage:
        def __init__(self) -> None:
            self.url = "about:blank"

        def set_default_timeout(self, timeout: int) -> None:
            events.append(("page_timeout", str(timeout)))

        def set_default_navigation_timeout(self, timeout: int) -> None:
            events.append(("nav_timeout", str(timeout)))

        def on(self, event_name: str, handler) -> None:
            events.append(("page_on", event_name))

        async def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
            self.url = url
            events.append(("goto", url))

        async def wait_for_load_state(self, load_state: str, timeout: int = 15000) -> None:
            events.append(("load_state", load_state))

        async def evaluate(self, script: str):
            return True

        async def close(self) -> None:
            events.append(("page_close", "1"))

    class FakeContext:
        def __init__(self, page: FakePage) -> None:
            self.pages = [page]

        async def close(self) -> None:
            events.append(("context_close", "1"))

    class FakeBrowser:
        def __init__(self, context: FakeContext) -> None:
            self.contexts = [context]

        async def close(self) -> None:
            events.append(("browser_close", "1"))

    class FakeChromium:
        def __init__(self, browser: FakeBrowser) -> None:
            self._browser = browser

        async def connect_over_cdp(self, connect_url: str, timeout: int):
            events.append(("connect_over_cdp", connect_url))
            events.append(("connect_timeout", str(timeout)))
            return self._browser

    class FakePlaywrightInstance:
        def __init__(self, browser: FakeBrowser) -> None:
            self.chromium = FakeChromium(browser)

        async def stop(self) -> None:
            events.append(("playwright_stop", "1"))

    class FakePlaywrightManager:
        def __init__(self, browser: FakeBrowser) -> None:
            self._browser = browser

        async def start(self):
            events.append(("playwright_start", "1"))
            return FakePlaywrightInstance(self._browser)

    fake_page = FakePage()
    fake_context = FakeContext(fake_page)
    fake_browser = FakeBrowser(fake_context)

    async def fake_create_session(self):
        events.append(("create_session", "1"))
        return {"id": "sess_123", "connectUrl": "wss://cdp.browserbase.example/session"}

    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: FakePlaywrightManager(fake_browser),
    )
    monkeypatch.setattr(LocalBrowserEnvironment, "_create_browserbase_session", fake_create_session)

    env = LocalBrowserEnvironment(
        browserbase_enabled=True,
        headless=True,
        slow_mo_ms=0,
        output_dir=tmp_path / "artifacts",
    )

    env.prepare(start_url="https://example.com", task="probe")

    assert ("create_session", "1") in events
    assert ("connect_over_cdp", "wss://cdp.browserbase.example/session") in events
    assert ("goto", "https://example.com") in events
    assert env.serialize()["environment"]["browserbase_session"]["id"] == "sess_123"

    env.close()


def test_local_browser_environment_retries_transient_browserbase_session_create(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}
    create_calls: list[dict[str, object]] = []

    class FakeSession:
        def __init__(self, session_id: str, connect_url: str) -> None:
            self.id = session_id
            self.connect_url = connect_url

    class FakeSessions:
        def create(self, **kwargs) -> FakeSession:
            create_calls.append(kwargs)
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise httpx.ConnectError("dns failed", request=httpx.Request("POST", "https://api.browserbase.com"))
            return FakeSession("sess_retry", "wss://cdp.browserbase.example/retry")

    class FakeBrowserbase:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.sessions = FakeSessions()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "browserbase", types.SimpleNamespace(Browserbase=FakeBrowserbase))
    monkeypatch.setattr("miniswewebagent.environments.local_browser.asyncio.to_thread", fake_to_thread)

    env = LocalBrowserEnvironment(
        browserbase_enabled=True,
        browserbase_api_key="key",
        browserbase_project_id="project",
        browser_navigation_timeout_ms=3000,
        browserbase_session_create_retries=3,
        browserbase_retry_backoff_seconds=0,
        output_dir=tmp_path / "artifacts",
    )

    session = env._run(env._create_browserbase_session())

    assert attempts["count"] == 3
    assert session["id"] == "sess_retry"
    assert create_calls[-1] == {
        "project_id": "project",
        "proxies": True,
        "browser_settings": {"advanced_stealth": True},
        "keep_alive": True,
        "timeout": 720,
    }


def test_capture_observation_recovers_after_closed_target(monkeypatch, tmp_path: Path) -> None:
    class FakeLocator:
        async def wait_for(self, state: str = "attached", timeout: int = 0) -> None:
            return None

        async def aria_snapshot(self) -> str:
            return "- document"

    class ClosedTargetError(RuntimeError):
        pass

    class FailingPage:
        url = "https://example.com/crashed"

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator()

        async def screenshot(self, path: str, full_page: bool = False) -> None:
            raise ClosedTargetError("Target page, context or browser has been closed")

        async def title(self) -> str:
            return ""

    class HealthyPage:
        url = "https://example.com/recovered"

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator()

        async def screenshot(self, path: str, full_page: bool = False) -> None:
            Path(path).write_bytes(b"png")

        async def title(self) -> str:
            return "Recovered"

    env = LocalBrowserEnvironment(
        observation_timeout_ms=2000,
        output_dir=tmp_path / "artifacts",
    )
    env._page = FailingPage()

    recovery_calls: list[tuple[str | None, str]] = []

    async def fake_recover_runtime(self, *, url_hint: str | None, source: str) -> bool:
        recovery_calls.append((url_hint, source))
        self._page = HealthyPage()
        return True

    monkeypatch.setattr(LocalBrowserEnvironment, "_recover_runtime", fake_recover_runtime)

    observation = env._run(env._capture_observation(success=False, exception_text=""))

    assert recovery_calls == [("https://example.com/crashed", "observation")]
    assert observation["screenshot_path"].endswith("step_0000.png")
    assert Path(observation["screenshot_path"]).exists()
    assert observation["title"] == "Recovered"
