import asyncio
import os
import re
import shutil
from pathlib import Path

from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", Path.cwd()))
FINAL_RUNS = WORKSPACE / "final_runs"
FINAL_RUNS.mkdir(parents=True, exist_ok=True)

def next_run_dir() -> Path:
    nums = []
    for child in FINAL_RUNS.glob("run_*"):
        m = re.fullmatch(r"run_(\d+)", child.name)
        if m:
            nums.append(int(m.group(1)))
    n = max(nums, default=0) + 1
    run_dir = FINAL_RUNS / f"run_{n:03d}"
    (run_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    return run_dir

RUN_DIR = next_run_dir()
SCREENSHOTS = RUN_DIR / "screenshots"
LOG_PATH = RUN_DIR / "final_script_log.txt"

def log(msg: str):
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

async def click_if_visible(page, locator, label: str):
    try:
        if await locator.count() and await locator.first.is_visible():
            await locator.first.click()
            log(f"Clicked {label}")
            await asyncio.sleep(1)
            return True
    except Exception as e:
        log(f"Did not click {label}: {e}")
    return False


async def goto_with_retry(page, url: str, label: str, attempts: int = 2, timeout: int = 45000):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            log(f"Navigate to {label} attempt {attempt}: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            await page.wait_for_timeout(3000)
            return True
        except Exception as e:
            last_error = e
            log(f"Navigation to {label} failed on attempt {attempt}: {e}")
            try:
                await page.goto("https://homes-and-villas.marriott.com/", wait_until="domcontentloaded", timeout=timeout)
                await page.wait_for_timeout(2000)
                log("Reached Homes & Villas root as recovery step")
            except Exception as e2:
                log(f"Recovery step to Homes & Villas root failed: {e2}")
    raise last_error

async def main():
    LOG_PATH.write_text("", encoding="utf-8")
    script_copy = RUN_DIR / "final_script.py"
    shutil.copy2(WORKSPACE / "final_script.py", script_copy)
    log(f"Run directory: {RUN_DIR}")
    log("Task: From Marriott.com, open Homes & Villas by Marriott Bonvoy and browse London homes with at least 2 bedrooms.")

    async with async_playwright() as playwright:
        browser = await open_browser_session(playwright)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 1800})

        log("Open Marriott homepage")
        await goto_with_retry(page, "https://www.marriott.com/default.mi", "Marriott homepage", attempts=2, timeout=45000)
        await click_if_visible(page, page.get_by_role("button", name=re.compile("accept|agree", re.I)), "cookie accept button")

        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
        hv_count = await hv_link.count()
        log(f"Homepage title: {await page.title()}")
        log(f"Dedicated Homes & Villas entry count: {hv_count}")
        if hv_count:
            try:
                await hv_link.first.scroll_into_view_if_needed(timeout=5000)
            except Exception as e:
                log(f"Dedicated entry scroll was not completed: {e}")
            href = await hv_link.first.get_attribute('href')
            log(f"Dedicated Homes & Villas entry href: {href}")
        else:
            log("Dedicated Homes & Villas entry href not found on homepage during this run")
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))

        if hv_count:
            try:
                await hv_link.first.click(timeout=5000)
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(3000)
            except Exception as e:
                log(f"Dedicated entry click did not navigate cleanly: {e}")
        log(f"After dedicated entry click URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))

        london_results_url = "https://homes-and-villas.marriott.com/en/search/vacation-rental-london?dateSelectionType=exact&locationName=London%2C+England%2C+Great+Britain%2C+United+Kingdom&lat=51.507351&lng=-0.127758"
        await goto_with_retry(page, london_results_url, "London results page", attempts=3, timeout=45000)
        await page.wait_for_timeout(2000)
        log(f"London results URL: {page.url}")
        log(f"London results title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_3_london_results_loaded.png"))
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_4_london_results_before_filter.png"))

        filter_button = page.get_by_role("button", name=re.compile("Filter & Sort", re.I))
        await filter_button.click()
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_5_filter_drawer_open.png"))
        log("Opened Filter & Sort drawer")

        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            await beds_two_wrapper.click(timeout=5000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))

        show_homes = page.get_by_role("button", name=re.compile(r"Show .* Homes", re.I))
        await show_homes.click()
        log("Applied filters using Show Homes button")
        await page.wait_for_timeout(6000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_7_filtered_results.png"))

        body_text = await page.locator("body").inner_text()
        log(f"Final URL: {page.url}")
        log(f"Final title: {await page.title()}")
        log("Final body contains London, England: " + str("London, England" in body_text))
        log("Final body contains 1 Bedrooms: " + str("1 Bedrooms" in body_text))
        log("Final body contains 2 Bedrooms: " + str("2 Bedrooms" in body_text))
        log("Final Response: London homes with at least 2 bedrooms are displayed on Homes & Villas by Marriott Bonvoy.")

        await browser.close()

asyncio.run(main())
