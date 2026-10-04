import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = repo_root / "docs" / "cli_actor_orchestrator_execution_flow.html"
    png_path = repo_root / "docs" / "cli_actor_orchestrator_execution_flow@2x.png"

    configured_browser = os.environ.get("MINIWEB_DIAGRAM_CHROMIUM", "").strip()
    browser_path = Path(configured_browser).expanduser() if configured_browser else None
    if browser_path is None:
        browser_cache = Path.home() / ".cache" / "ms-playwright"
        candidates = sorted(browser_cache.glob("chromium-*/chrome-linux64/chrome"))
        browser_path = candidates[-1] if candidates else None

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(browser_path) if browser_path is not None else None,
        )
        page = browser.new_page(
            viewport={"width": 1900, "height": 1200},
            device_scale_factor=2,
        )
        page.goto(html_path.as_uri(), wait_until="load")
        page.locator("#capture").screenshot(
            path=str(png_path),
            type="png",
            animations="disabled",
        )
        browser.close()

    print(f"Rendered {png_path}")


if __name__ == "__main__":
    main()
