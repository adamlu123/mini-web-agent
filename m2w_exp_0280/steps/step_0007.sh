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
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_hv_home.png'))
        dest = page.locator('input[type="text"]').first
        await dest.click()
        await dest.fill('London')
        await asyncio.sleep(2)
        await page.screenshot(path=str(SS/'explore_hv_london_suggestions.png'))
        print('URL1', page.url)
        print('TITLE1', await page.title())
        print('BODY1', (await page.locator('body').inner_text())[:3000])
        suggestions = await page.locator('li,[role="option"],button,a,div').evaluate_all("els => els.map(e => ({t:(e.innerText||'').trim(), r:e.getAttribute('role')||'', c:e.className||''})).filter(x => /London/i.test(x.t)).slice(0,30)")
        print('LONDON_MATCHES', json.dumps(suggestions, indent=2)[:4000])
        try:
            await page.get_by_text('London', exact=False).first.click(timeout=5000)
            print('CLICKED suggestion')
        except Exception as e:
            print('SUGGESTION CLICK FAIL', e)
        await asyncio.sleep(1)
        try:
            await page.get_by_role('button', name='Search').click(timeout=5000)
        except Exception as e:
            print('SEARCH CLICK FAIL', e)
        await page.wait_for_load_state('domcontentloaded')
        await asyncio.sleep(3)
        await page.screenshot(path=str(SS/'explore_hv_results_london.png'))
        print('URL2', page.url)
        print('TITLE2', await page.title())
        print('BODY2', (await page.locator('body').inner_text())[:5000])
        controls = await page.locator('button,input,select,[role="button"],[role="checkbox"],[role="radio"],[aria-label]').evaluate_all("els => els.map(e => ({tag:e.tagName, text:(e.innerText||e.value||e.getAttribute('aria-label')||'').trim(), role:e.getAttribute('role')||'', aria:e.getAttribute('aria-label')||'', type:e.getAttribute('type')||''})).filter(x => /bed|bedroom|guest|filter|sort|London/i.test((x.text+' '+x.aria))).slice(0,80)")
        print('CONTROLS', json.dumps(controls, indent=2)[:5000])
        await browser.close()

asyncio.run(main())
PY
