python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.environ['WORKSPACE_DIR'])
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as pw:
        browser = await open_browser_session(pw)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_home.png'))
        href = await page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]').first.get_attribute('href')
        print('HREF', href)
        await page.goto(href, wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_collections.png'))
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        tb = page.locator('input[type="text"]').first
        await tb.fill('London')
        await page.get_by_text('London, England, Great Britain, United Kingdom', exact=True).click()
        await page.locator('button[type="submit"]').first.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.screenshot(path=str(SS/'explore_london_results.png'))
        print('RESULT_URL', page.url)
        print('RESULT_TITLE', await page.title())
        await page.get_by_role('button', name='Filter & Sort').click()
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(SS/'explore_filter_drawer.png'))
        print('DRAWER_TEXT_START')
        txt = await page.locator('body').inner_text()
        idx = txt.find('Filter')
        print(txt[idx:idx+5000])
        print('DRAWER_TEXT_END')
        print('DRAWER_ARIA_START')
        print(await page.locator('body').aria_snapshot())
        print('DRAWER_ARIA_END')
        await browser.close()

asyncio.run(main())
PY
