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
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://www.marriott.com/', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_home.png'))
        print('URL1', page.url)
        print('TITLE1', await page.title())
        sect = page.locator("text=Vacation Home Rentals").first
        print('SECTION COUNT', await page.locator("text=Vacation Home Rentals").count())
        if await sect.count():
            await sect.scroll_into_view_if_needed()
            await page.screenshot(path=str(SS/'explore_vacation_section.png'))
            print('SECTION TEXT', await sect.locator('xpath=..').inner_text())
        links = page.locator('a')
        c = await links.count()
        vals = []
        for i in range(min(c, 120)):
            try:
                txt = (await links.nth(i).inner_text()).strip()
                href = await links.nth(i).get_attribute('href')
                if txt or href:
                    vals.append((txt, href))
            except:
                pass
        for item in vals:
            if 'villa' in str(item).lower() or 'home' in str(item).lower() or 'rental' in str(item).lower():
                print('LINK', item)
        clicked = False
        for text in ['Vacation Home Rentals', 'View More', 'Homes & Villas', 'Rental Homes with Incredible Pools']:
            try:
                loc = page.get_by_text(text, exact=False).first
                if await loc.count():
                    await loc.scroll_into_view_if_needed()
                    print('TRY CLICK', text)
                    await loc.click(timeout=5000)
                    clicked = True
                    break
            except Exception as e:
                print('CLICKFAIL', text, e)
        await page.wait_for_timeout(4000)
        print('URL2', page.url)
        print('TITLE2', await page.title())
        await page.screenshot(path=str(SS/'explore_after_click.png'))
        print('BODY2', (await page.locator('body').inner_text())[:5000])
        await browser.close()

asyncio.run(main())
PY
