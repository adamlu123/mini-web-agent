python - <<'PY'
import asyncio, re, os
from playwright.async_api import async_playwright
from browser_session import open_browser_session

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 1800})
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.wait_for_timeout(5000)
        for loc in [
            page.get_by_role('button', name=re.compile('accept|agree', re.I)),
            page.get_by_role('button', name=re.compile('close', re.I)),
        ]:
            try:
                if await loc.count() and await loc.first.is_visible():
                    await loc.first.click(timeout=2000)
                    await page.wait_for_timeout(1000)
            except Exception:
                pass
        print('URL', page.url)
        print('TITLE', await page.title())
        textboxes = page.get_by_role('textbox')
        print('TEXTBOX_COUNT', await textboxes.count())
        for i in range(await textboxes.count()):
            tb = textboxes.nth(i)
            try:
                print('TB', i, await tb.evaluate("e => ({outer:e.outerHTML, aria:e.getAttribute('aria-label'), placeholder:e.getAttribute('placeholder'), name:e.getAttribute('name'), id:e.id, type:e.type})"))
            except Exception as e:
                print('TBERR', i, e)
        candidates = [
            page.get_by_role('textbox', name=re.compile('destination|search destination', re.I)),
            page.locator('input[placeholder*="Destination" i]'),
            page.locator('input[aria-label*="destination" i]'),
            page.locator('input[name*="destination" i]'),
            page.locator('input:not([type="hidden"])').filter(has=page.locator('xpath=..')),
        ]
        for idx, loc in enumerate(candidates):
            try:
                print('CAND', idx, 'count', await loc.count())
            except Exception as e:
                print('CANDERR', idx, e)
        await browser.close()

asyncio.run(main())
PY
