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
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.locator('input[type="text"]').first.fill('London')
        await page.wait_for_timeout(1500)
        suggestion = page.get_by_text('London, England, Great Britain, United Kingdom', exact=True)
        print('SUGGESTION COUNT', await suggestion.count())
        if await suggestion.count():
            await suggestion.first.click()
        else:
            await page.keyboard.press('ArrowDown')
            await page.keyboard.press('Enter')
        await page.wait_for_timeout(1000)
        print('AFTER SELECT URL', page.url)
        print('AFTER SELECT BODY', (await page.locator('body').inner_text())[:1500])
        await page.screenshot(path=str(SS/'after_select_london.png'))
        btn = page.get_by_role('button', name='Search')
        print('SEARCH ENABLED', await btn.is_enabled())
        await btn.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(5000)
        print('RESULT URL', page.url)
        print('TITLE', await page.title())
        body=(await page.locator('body').inner_text())[:4000]
        print('BODY', body)
        texts=[]
        for sel in ['button','a','input','label','div','span']:
            loc=page.locator(sel)
            n=min(await loc.count(),120)
            for i in range(n):
                try:
                    t=(await loc.nth(i).inner_text()).strip()
                except:
                    t=''
                if any(k in t.lower() for k in ['bed','guest','filter','sort','london','bedroom']):
                    texts.append((sel,t[:200]))
        print(json.dumps(texts[:80], indent=2))
        await page.screenshot(path=str(SS/'results_london.png'))
        await browser.close()

asyncio.run(main())
PY
