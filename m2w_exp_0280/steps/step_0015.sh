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
        await page.set_viewport_size({"width": 1280, "height": 1800})
        await page.goto('https://homes-and-villas.marriott.com/en/collections', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'collections_page.png'))
        print('URL', page.url)
        print('TITLE', await page.title())
        print('TEXTBOX_COUNT', await page.get_by_role('textbox').count())
        for i in range(await page.get_by_role('textbox').count()):
            try:
                tb = page.get_by_role('textbox').nth(i)
                print('TB', i, await tb.get_attribute('placeholder'), await tb.get_attribute('aria-label'))
            except Exception as e:
                print('TB_ERR', i, e)
        links = page.locator('a[href]')
        cnt = await links.count()
        for i in range(min(cnt, 80)):
            href = await links.nth(i).get_attribute('href')
            txt = (await links.nth(i).inner_text()).strip().replace('\n',' ')[:120]
            if href and 'homes-and-villas.marriott.com' in href:
                print('HVLINK', i, txt, href)
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'hv_root_page.png'))
        print('ROOT_URL', page.url)
        print('ROOT_TITLE', await page.title())
        print('ROOT_TEXTBOX_COUNT', await page.get_by_role('textbox').count())
        for i in range(await page.get_by_role('textbox').count()):
            tb = page.get_by_role('textbox').nth(i)
            print('ROOT_TB', i, await tb.get_attribute('placeholder'), await tb.get_attribute('aria-label'))
        await browser.close()

asyncio.run(main())
PY
