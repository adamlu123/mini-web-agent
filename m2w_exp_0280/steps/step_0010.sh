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
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.locator('input[type="text"]').first.fill('London')
        await asyncio.sleep(2)
        print('INPUT VALUE', await page.locator('input[type="text"]').first.input_value())
        print('ARIA BEFORE', await page.locator('body').aria_snapshot())
        suggestions = page.locator('text=London, England, Great Britain, United Kingdom')
        print('MATCHES', await suggestions.count())
        for i in range(await suggestions.count()):
            try:
                print('TXT', i, await suggestions.nth(i).inner_text())
            except Exception as e:
                print('TXTERR', i, e)
        await page.screenshot(path=str(SS/'explore_london_suggestions2.png'))
        try:
            await suggestions.last.click(timeout=5000)
            print('CLICKED SUGGESTION')
        except Exception as e:
            print('CLICK ERR', e)
        await asyncio.sleep(1)
        print('VALUE AFTER CLICK', await page.locator('input[type="text"]').first.input_value())
        submit = page.locator('button[type="submit"]').first
        print('SUBMIT COUNT', await page.locator('button[type="submit"]').count())
        print('SUBMIT TEXT', await submit.inner_text())
        print('SUBMIT DISABLED', await submit.is_disabled())
        try:
            await submit.click(timeout=5000)
            print('CLICKED SUBMIT')
        except Exception as e:
            print('SUBMIT ERR', e)
        await asyncio.sleep(5)
        print('URL AFTER SUBMIT', page.url)
        print('TITLE AFTER SUBMIT', await page.title())
        print('BODY AFTER SUBMIT', (await page.locator('body').inner_text())[:5000])
        await page.screenshot(path=str(SS/'explore_after_london_submit.png'))
        await browser.close()

asyncio.run(main())
PY
