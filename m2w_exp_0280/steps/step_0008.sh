python - <<'PY'
import asyncio, os, json
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE=Path(os.getcwd())
SS=WORKSPACE/'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser=await open_browser_session(p)
        context=browser.contexts[0] if browser.contexts else await browser.new_context()
        page=context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_hv_home.png'))
        loc=page.locator('input[aria-label="DESTINATION"], input[type="text"]').first
        print('INPUT COUNT', await page.locator('input').count())
        print('ARIA SNAP', await page.locator('form').first.aria_snapshot())
        print('INPUT BOX', await loc.bounding_box())
        print('INPUT VALUE BEFORE', await loc.input_value())
        try:
            await loc.fill('London')
            print('FILL OK')
        except Exception as e:
            print('FILL FAIL', e)
        await page.wait_for_timeout(2000)
        print('INPUT VALUE AFTER', await loc.input_value())
        items=page.locator('[role="option"], li, [data-testid*="option"], [class*="suggest"], [class*="typeahead"] li')
        texts=[]
        for i in range(min(await items.count(),20)):
            try:
                t=(await items.nth(i).inner_text()).strip()
                if t:
                    texts.append(t)
            except: pass
        print('ITEMS', texts)
        print('BODY', (await page.locator('body').inner_text())[:3000])
        await page.screenshot(path=str(SS/'explore_hv_london_typed.png'))
        await browser.close()

asyncio.run(main())
PY
