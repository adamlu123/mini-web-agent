python - <<'PY'
import asyncio, os, json
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.getcwd())
( WORKSPACE / 'screenshots').mkdir(exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://www.marriott.com/', wait_until='domcontentloaded')
        await page.wait_for_timeout(5000)
        await page.screenshot(path=str(WORKSPACE/'screenshots'/'explore_home_links.png'))
        js = """
        () => Array.from(document.querySelectorAll('a,button')).map(el => ({
          tag: el.tagName,
          text: (el.innerText||el.getAttribute('aria-label')||'').trim().replace(/\s+/g,' '),
          href: el.href || '',
          role: el.getAttribute('role') || ''
        })).filter(x => /homes|villas|vacation|view more/i.test(x.text) || /homes|villas/i.test(x.href))
        """
        items = await page.evaluate(js)
        print(json.dumps(items[:80], indent=2))
        await browser.close()

asyncio.run(main())
PY
