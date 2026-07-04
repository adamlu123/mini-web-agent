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
        await page.set_viewport_size({"width":1280,"height":1800})
        url='https://homes-and-villas.marriott.com/en/search/vacation-rental-london?dateSelectionType=exact&locationName=London%2C+England%2C+Great+Britain%2C+United+Kingdom&lat=51.507351&lng=-0.127758'
        await page.goto(url, wait_until='domcontentloaded')
        await page.wait_for_timeout(5000)
        fb = page.get_by_role('button', name=re.compile('Filter & Sort', re.I))
        await fb.click()
        await page.wait_for_timeout(3000)
        print('TITLE', await page.title())
        print('DRAWER_TEXT_START')
        txt = await page.locator('body').inner_text()
        i = txt.find('Beds & Baths')
        print(txt[i:i+1500] if i!=-1 else txt[:2000])
        print('DRAWER_TEXT_END')
        print('BUTTONS_START')
        buttons = page.get_by_role('button')
        n = await buttons.count()
        for idx in range(min(n,120)):
            b = buttons.nth(idx)
            try:
                name = await b.inner_text()
                aria = await b.get_attribute('aria-label')
                pressed = await b.get_attribute('aria-pressed')
                checked = await b.get_attribute('aria-checked')
                cls = await b.get_attribute('class')
                if any(s and '2' in s for s in [name, aria, cls]) or any(s and 'Bedrooms' in s for s in [name, aria, cls]):
                    print('BTN', idx, {'name':name, 'aria':aria, 'pressed':pressed, 'checked':checked, 'class':cls})
            except Exception:
                pass
        print('BUTTONS_END')
        print('TEXT_2_COUNT', await page.get_by_text('2', exact=True).count())
        print('TEXT_BEDROOMS_COUNT', await page.get_by_text('Bedrooms', exact=True).count())
        await browser.close()

asyncio.run(main())
PY
