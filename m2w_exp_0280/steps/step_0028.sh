python - <<'PY'
import asyncio, os, re
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 1800})
        url = "https://homes-and-villas.marriott.com/en/search/vacation-rental-london?dateSelectionType=exact&locationName=London%2C+England%2C+Great+Britain%2C+United+Kingdom&lat=51.507351&lng=-0.127758"
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        btn = page.get_by_role("button", name=re.compile("Filter & Sort", re.I))
        await btn.click()
        await page.wait_for_timeout(2500)
        print("URL", page.url)
        dialogs = page.get_by_role('dialog')
        print('DIALOG_COUNT', await dialogs.count())
        for i in range(await dialogs.count()):
            d = dialogs.nth(i)
            try:
                if await d.is_visible():
                    txt = await d.inner_text()
                    print('DIALOG', i, txt[:2000])
            except Exception as e:
                print('DIALOG_ERR', i, e)
        drawers = page.locator('div').filter(has_text='Beds & Baths')
        print('DIV_HAS_BEDS_BATHS_COUNT', await drawers.count())
        for i in range(min(await drawers.count(), 5)):
            el = drawers.nth(i)
            try:
                if await el.is_visible():
                    print('DIV', i, (await el.inner_text())[:1500])
                    print('HTML', i, (await el.evaluate("e => e.outerHTML"))[:3000])
            except Exception as e:
                print('DIV_ERR', i, e)
        buttons2 = page.get_by_role('button', name='2')
        print('BUTTON2_COUNT', await buttons2.count())
        for i in range(await buttons2.count()):
            b = buttons2.nth(i)
            try:
                vis = await b.is_visible()
                txt = await b.text_content()
                if vis:
                    print('BUTTON2_VISIBLE', i, txt, await b.evaluate("e => e.outerHTML")[:1000])
            except Exception as e:
                print('BUTTON2_ERR', i, e)
        await browser.close()

asyncio.run(main())
PY
