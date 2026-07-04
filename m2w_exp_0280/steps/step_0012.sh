python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.getcwd())
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 1800})
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS / 'explore_home.png'))
        view_more = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]').first
        print('VIEW_MORE_COUNT', await page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]').count())
        await view_more.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.screenshot(path=str(SS / 'explore_hv_landing.png'))
        tb = page.get_by_role('textbox', name='Search destination, landmark, address').first
        await tb.fill('London')
        await page.get_by_text('London, England, Great Britain, United Kingdom', exact=True).click()
        await page.screenshot(path=str(SS / 'explore_london_selected.png'))
        await page.locator('button[type="submit"]').first.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.screenshot(path=str(SS / 'explore_results.png'))
        print('RESULT_URL', page.url)
        print('RESULT_TITLE', await page.title())
        fs = page.get_by_role('button', name='Filter & Sort').first
        await fs.click()
        await asyncio.sleep(2)
        await page.screenshot(path=str(SS / 'explore_filter_drawer.png'))
        txt = await page.locator('body').inner_text()
        print('BODY_START')
        print(txt[:12000])
        print('BODY_END')
        print('ARIA_START')
        print(await page.locator('body').aria_snapshot())
        print('ARIA_END')
        await browser.close()

asyncio.run(main())
PY
