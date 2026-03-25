from __future__ import annotations

from pathlib import Path

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
