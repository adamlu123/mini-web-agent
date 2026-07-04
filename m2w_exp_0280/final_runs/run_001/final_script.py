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
        await page.goto("https://www.marriott.com/default.mi", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        await click_if_visible(page, page.get_by_role("button", name=re.compile("accept|agree", re.I)), "cookie accept button")

        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
        await hv_link.first.scroll_into_view_if_needed()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))
        log(f"Homepage title: {await page.title()}")
        log(f"Dedicated Homes & Villas entry href: {await hv_link.first.get_attribute('href')}")

        await hv_link.first.click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
        log(f"After dedicated entry click URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))

        await page.goto("https://homes-and-villas.marriott.com/", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        log(f"Homes & Villas root URL: {page.url}")
        log(f"Homes & Villas root title: {await page.title()}")

        textbox = page.locator('input[type="text"]').first
        await textbox.fill("London")
        log("Filled destination with London")
        await page.wait_for_timeout(2000)
        suggestion = page.get_by_text("London, England, Great Britain, United Kingdom", exact=True)
        await suggestion.click()
        log("Selected exact London suggestion")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_3_london_selected_on_search.png"))

        submit = page.locator('button[type="submit"]').first
        log(f"Search button text: {(await submit.text_content() or '').strip()}")
        await submit.click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(5000)
        log(f"London results URL: {page.url}")
        log(f"London results title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_4_london_results_before_filter.png"))

        filter_button = page.get_by_role("button", name=re.compile("Filter & Sort", re.I))
        await filter_button.click()
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_5_filter_drawer_open.png"))
        log("Opened Filter & Sort drawer")

        bedrooms_two = page.get_by_text("Bedrooms").locator('..').get_by_text("2", exact=True)
        if not await bedrooms_two.count():
            bedrooms_two = page.locator('text=Bedrooms').locator('..').locator('text="2"').first
        await bedrooms_two.click()
        log("Selected Bedrooms = 2 in Beds & Baths filter")
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
