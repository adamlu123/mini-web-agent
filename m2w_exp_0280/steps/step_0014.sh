python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.environ['WORKSPACE_DIR'])
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_home.png'))
        await page.goto('https://homes-and-villas.marriott.com/en/collections', wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)
        print('COLLECTIONS_URL', page.url)
        print('COLLECTIONS_TITLE', await page.title())
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)
        tb = page.get_by_role('textbox', name='Search destination, landmark, address').first
        await tb.fill('London')
        await page.wait_for_timeout(2000)
        await page.get_by_text('London, England, Great Britain, United Kingdom', exact=True).click()
        await page.wait_for_timeout(1000)
        await page.locator('button[type="submit"]').first.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(5000)
        print('RESULTS_URL', page.url)
        print('RESULTS_TITLE', await page.title())
        await page.screenshot(path=str(SS/'explore_results_london.png'))
        fs = page.get_by_role('button', name='Filter & Sort').first
        print('FILTER_SORT_COUNT', await page.get_by_role('button', name='Filter & Sort').count())
        await fs.click()
        await page.wait_for_timeout(3000)
        await page.screenshot(path=str(SS/'explore_filter_drawer.png'))
        body = await page.locator('body').inner_text()
        print('DRAWER_TEXT_START')
        print(body[:6000])
        print('DRAWER_TEXT_END')
        print('DRAWER_ARIA_START')
        print(await page.locator('body').aria_snapshot())
        print('DRAWER_ARIA_END')
        await browser.close()

asyncio.run(main())
PY
