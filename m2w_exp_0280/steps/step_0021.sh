python - <<'PY'
import asyncio, re, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

async def main():
    async with async_playwright() as playwright:
        browser = await open_browser_session(playwright)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 1800})
        await page.goto("https://homes-and-villas.marriott.com/", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        textbox = page.locator('input[type="text"]').first
        await textbox.fill('London')
        await page.wait_for_timeout(2000)
        candidates = [
            page.get_by_role('option', name=re.compile(r'^London, England, Great Britain, United Kingdom$', re.I)),
            page.locator('[role="option"]').filter(has_text='London, England, Great Britain, United Kingdom'),
            page.get_by_text('London, England, Great Britain, United Kingdom', exact=True),
            page.locator('li,div,span').filter(has_text='London, England, Great Britain, United Kingdom').first,
        ]
        for c in candidates:
            try:
                if await c.count() and await c.first.is_visible():
                    await c.first.click()
                    break
            except Exception:
                pass
        await page.wait_for_timeout(1200)
        await page.locator('button[type="submit"]').first.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(5000)
        await page.get_by_role('button', name=re.compile('Filter & Sort', re.I)).click()
        await page.wait_for_timeout(2500)
        print('TITLE:', await page.title())
        print('URL:', page.url)
        body = await page.locator('body').inner_text()
        idx = body.find('Bedrooms')
        print('BODY_SNIPPET:', body[idx:idx+1200] if idx != -1 else body[:2000])
        print('BUTTONS:')
        for i, b in enumerate(await page.get_by_role('button').all()):
            try:
                txt = (await b.text_content() or '').strip().replace('\n',' ')
                if txt and any(k in txt for k in ['Bedrooms','Beds','Baths','Any','Show','Filter','2']):
                    print(i, repr(txt))
            except Exception:
                pass
        print('ARIA_BODY:')
        print(await page.locator('body').aria_snapshot())
        await browser.close()

asyncio.run(main())
PY
