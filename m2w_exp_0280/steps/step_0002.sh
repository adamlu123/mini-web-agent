python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.getcwd())
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def dismiss(page):
    for name in ['Close', 'close', 'Dismiss', 'No Thanks', 'Accept', 'Continue', 'Got it']:
        try:
            btn = page.get_by_role('button', name=name)
            if await btn.count():
                await btn.first.click(timeout=2000)
                await page.wait_for_timeout(1000)
        except:
            pass

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({'width':1280,'height':1800})
        await page.goto('https://www.marriott.com/', wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)
        await dismiss(page)
        await page.screenshot(path=str(SS/'explore_home.png'))
        print('HOME URL', page.url)
        print('HOME TITLE', await page.title())
        print('LINKS', await page.locator('a').evaluate_all("els => els.slice(0,80).map(e => (e.innerText||e.getAttribute('aria-label')||'').trim()).filter(Boolean)"))
        text = await page.locator('body').inner_text()
        for needle in ['Homes', 'Villas', 'Homes & Villas', 'Marriott Bonvoy']:
            print('HAS', needle, needle.lower() in text.lower())
        try:
            await page.get_by_role('link', name=lambda s: s and 'Homes & Villas' in s).first.click(timeout=5000)
        except Exception as e:
            print('CLICK FAIL 1', e)
            try:
                await page.locator('a:has-text("Homes & Villas")').first.click(timeout=5000)
            except Exception as e2:
                print('CLICK FAIL 2', e2)
        await page.wait_for_timeout(5000)
        await page.screenshot(path=str(SS/'explore_after_click.png'))
        print('AFTER URL', page.url)
        print('AFTER TITLE', await page.title())
        print('BODY SAMPLE', (await page.locator('body').inner_text())[:4000])
        await browser.close()

asyncio.run(main())
PY
