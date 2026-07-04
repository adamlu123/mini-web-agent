python - <<'PY'
import asyncio, os, json
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
        await page.set_viewport_size({"width": 1280, "height": 1800})
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.wait_for_timeout(4000)
        print('URL', page.url)
        print('TITLE', await page.title())
        txt = await page.locator('body').inner_text()
        print('BODY', txt[:4000])
        data = await page.evaluate('''() => Array.from(document.querySelectorAll('button,input,[role="button"],[role="combobox"],a')).map(el => ({tag: el.tagName, text: (el.innerText||el.value||el.getAttribute('aria-label')||'').trim(), role: el.getAttribute('role')||'', type: el.getAttribute('type')||''})).filter(x => /where|destination|search|bed|bedroom|guest|location|filter|london|check|home|villa/i.test(x.text)).slice(0,200)''')
        print(json.dumps(data, indent=2))
        await page.screenshot(path=str(SS/'homes_villas_home.png'))
        await browser.close()

asyncio.run(main())
PY
