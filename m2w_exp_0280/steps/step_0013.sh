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
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded', timeout=60000)
        await page.screenshot(path=str(SS/'inspect_homepage.png'))
        body = await page.locator('body').inner_text()
        print('BODY_SNIPPET_START')
        idx = body.find('Vacation Home Rentals')
        print(body[idx:idx+1200] if idx!=-1 else body[:1200])
        print('BODY_SNIPPET_END')
        links = page.locator('a')
        count = await links.count()
        for i in range(count):
            a = links.nth(i)
            try:
                href = await a.get_attribute('href')
                txt = (await a.inner_text()).strip().replace('\n',' ')
                if href and 'homes-and-villas.marriott.com' in href:
                    print('HV_LINK', i, 'TEXT=', txt[:200], 'HREF=', href)
            except Exception:
                pass
        print('ARIA_START')
        print((await page.locator('body').aria_snapshot())[:6000])
        print('ARIA_END')
        await browser.close()

asyncio.run(main())
PY
