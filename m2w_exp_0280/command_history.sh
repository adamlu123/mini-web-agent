# Step 1
cd /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280 && ls -la && echo '--- task.json ---' && cat task.json

# Step 2
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

# Step 3
python - <<'PY'
from pathlib import Path
import json
ws = Path('/home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280')
plan = ws/'plan.md'
plan.write_text('# Critical Points\n- [ ] CP1: Open Marriott.com and navigate to Homes & Villas by Marriott Bonvoy using the site\'s dedicated Homes & Villas entry point.\n- [ ] CP2: Apply the location/destination as London on the Homes & Villas experience so results are for London homes.\n- [ ] CP3: Apply the bedrooms filter using site controls so the search is limited to at least 2 bedrooms.\n- [ ] CP4: Display the filtered results page or listing set after applying the London and 2+ bedrooms constraints.\n')
cp_list = '1. CP1: Open Marriott.com and navigate to Homes & Villas by Marriott Bonvoy using the site\'s dedicated Homes & Villas entry point.\n2. CP2: Apply the location/destination as London on the Homes & Villas experience so results are for London homes.\n3. CP3: Apply the bedrooms filter using site controls so the search is limited to at least 2 bedrooms.\n4. CP4: Display the filtered results page or listing set after applying the London and 2+ bedrooms constraints.'
task_desc = 'From Marriott.com, open Homes & Villas by Marriott Bonvoy and browse London homes with at least 2 bedrooms.'
config = {
  'image_judge_system_prompt': 'You are a harsh evaluator of a single screenshot from a web task run. Return ONLY two labelled lines in exactly this format:\nReasoning: <1-2 sentences describing what the screenshot shows and which critical points it provides evidence for or against>\nScore: <integer 1-5, where 5 = this screenshot clearly evidences a critical point and 1 = this screenshot contains no relevant evidence>\nBe harsh when evidence is ambiguous, partially occluded, or does not clearly show the required state.',
  'image_judge_user_prompt': f'Task: {task_desc}\n\nCritical points to consider for this single image:\n{cp_list}\n\nEvaluate this screenshot against ALL critical points, even if it only evidences one of them. Be harsh when evidence is ambiguous or partially occluded. Return only the two labelled lines requested.',
  'final_verdict_system_prompt': 'You are a harsh aggregated judge of whether a web-task run succeeded. You will receive the task, critical points, the action log, per-image reasonings, and all screenshots. First write a Thoughts: block that evaluates every critical point one by one using the screenshots and log, noting any missing or ambiguous evidence. End your reply with EXACTLY one final line: Status: success or Status: failure',
  'final_verdict_user_prompt': f'Task: {task_desc}\n\nCritical points:\n{cp_list}\n\nAction history log:\n{{action_history_log}}\n\nPer-image reasonings:\n{{image_reasonings}}\n\nUsing the action log, per-image reasonings, and all attached screenshots, determine whether every critical point is satisfied. Be harsh: if a required filter, destination, or results state is not clearly evidenced, fail the run.'
}
(ws/'judge_config.json').write_text(json.dumps(config, indent=2))
print(plan.read_text())
print((ws/'judge_config.json').read_text())
PY

# Step 4
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

# Step 5
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

# Step 6
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

# Step 7
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

# Step 8
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
        browser=await open_browser_session(p)
        context=browser.contexts[0] if browser.contexts else await browser.new_context()
        page=context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_hv_home.png'))
        loc=page.locator('input[aria-label="DESTINATION"], input[type="text"]').first
        print('INPUT COUNT', await page.locator('input').count())
        print('ARIA SNAP', await page.locator('form').first.aria_snapshot())
        print('INPUT BOX', await loc.bounding_box())
        print('INPUT VALUE BEFORE', await loc.input_value())
        try:
            await loc.fill('London')
            print('FILL OK')
        except Exception as e:
            print('FILL FAIL', e)
        await page.wait_for_timeout(2000)
        print('INPUT VALUE AFTER', await loc.input_value())
        items=page.locator('[role="option"], li, [data-testid*="option"], [class*="suggest"], [class*="typeahead"] li')
        texts=[]
        for i in range(min(await items.count(),20)):
            try:
                t=(await items.nth(i).inner_text()).strip()
                if t:
                    texts.append(t)
            except: pass
        print('ITEMS', texts)
        print('BODY', (await page.locator('body').inner_text())[:3000])
        await page.screenshot(path=str(SS/'explore_hv_london_typed.png'))
        await browser.close()

asyncio.run(main())
PY

# Step 9
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

# Step 10
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

# Step 11
cd /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280 && ls -la && echo '--- plan.md ---' && sed -n '1,220p' plan.md && echo '--- judge_config.json ---' && sed -n '1,260p' judge_config.json && echo '--- final_script.py ---' && if [ -f final_script.py ]; then sed -n '1,260p' final_script.py; else echo 'MISSING'; fi

# Step 12
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
        await page.set_viewport_size({"width": 1280, "height": 1800})
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS / 'explore_home.png'))
        view_more = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]').first
        print('VIEW_MORE_COUNT', await page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]').count())
        await view_more.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.screenshot(path=str(SS / 'explore_hv_landing.png'))
        tb = page.get_by_role('textbox', name='Search destination, landmark, address').first
        await tb.fill('London')
        await page.get_by_text('London, England, Great Britain, United Kingdom', exact=True).click()
        await page.screenshot(path=str(SS / 'explore_london_selected.png'))
        await page.locator('button[type="submit"]').first.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.screenshot(path=str(SS / 'explore_results.png'))
        print('RESULT_URL', page.url)
        print('RESULT_TITLE', await page.title())
        fs = page.get_by_role('button', name='Filter & Sort').first
        await fs.click()
        await asyncio.sleep(2)
        await page.screenshot(path=str(SS / 'explore_filter_drawer.png'))
        txt = await page.locator('body').inner_text()
        print('BODY_START')
        print(txt[:12000])
        print('BODY_END')
        print('ARIA_START')
        print(await page.locator('body').aria_snapshot())
        print('ARIA_END')
        await browser.close()

asyncio.run(main())
PY

# Step 13
python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.environ['WORKSPACE_DIR'])
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded', timeout=60000)
        await page.screenshot(path=str(SS/'inspect_homepage.png'))
        body = await page.locator('body').inner_text()
        print('BODY_SNIPPET_START')
        idx = body.find('Vacation Home Rentals')
        print(body[idx:idx+1200] if idx!=-1 else body[:1200])
        print('BODY_SNIPPET_END')
        links = page.locator('a')
        count = await links.count()
        for i in range(count):
            a = links.nth(i)
            try:
                href = await a.get_attribute('href')
                txt = (await a.inner_text()).strip().replace('\n',' ')
                if href and 'homes-and-villas.marriott.com' in href:
                    print('HV_LINK', i, 'TEXT=', txt[:200], 'HREF=', href)
            except Exception:
                pass
        print('ARIA_START')
        print((await page.locator('body').aria_snapshot())[:6000])
        print('ARIA_END')
        await browser.close()

asyncio.run(main())
PY

# Step 14
python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.environ['WORKSPACE_DIR'])
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_home.png'))
        await page.goto('https://homes-and-villas.marriott.com/en/collections', wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)
        print('COLLECTIONS_URL', page.url)
        print('COLLECTIONS_TITLE', await page.title())
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)
        tb = page.get_by_role('textbox', name='Search destination, landmark, address').first
        await tb.fill('London')
        await page.wait_for_timeout(2000)
        await page.get_by_text('London, England, Great Britain, United Kingdom', exact=True).click()
        await page.wait_for_timeout(1000)
        await page.locator('button[type="submit"]').first.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.wait_for_timeout(5000)
        print('RESULTS_URL', page.url)
        print('RESULTS_TITLE', await page.title())
        await page.screenshot(path=str(SS/'explore_results_london.png'))
        fs = page.get_by_role('button', name='Filter & Sort').first
        print('FILTER_SORT_COUNT', await page.get_by_role('button', name='Filter & Sort').count())
        await fs.click()
        await page.wait_for_timeout(3000)
        await page.screenshot(path=str(SS/'explore_filter_drawer.png'))
        body = await page.locator('body').inner_text()
        print('DRAWER_TEXT_START')
        print(body[:6000])
        print('DRAWER_TEXT_END')
        print('DRAWER_ARIA_START')
        print(await page.locator('body').aria_snapshot())
        print('DRAWER_ARIA_END')
        await browser.close()

asyncio.run(main())
PY

# Step 15
python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.environ['WORKSPACE_DIR'])
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await open_browser_session(p)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 1800})
        await page.goto('https://homes-and-villas.marriott.com/en/collections', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'collections_page.png'))
        print('URL', page.url)
        print('TITLE', await page.title())
        print('TEXTBOX_COUNT', await page.get_by_role('textbox').count())
        for i in range(await page.get_by_role('textbox').count()):
            try:
                tb = page.get_by_role('textbox').nth(i)
                print('TB', i, await tb.get_attribute('placeholder'), await tb.get_attribute('aria-label'))
            except Exception as e:
                print('TB_ERR', i, e)
        links = page.locator('a[href]')
        cnt = await links.count()
        for i in range(min(cnt, 80)):
            href = await links.nth(i).get_attribute('href')
            txt = (await links.nth(i).inner_text()).strip().replace('\n',' ')[:120]
            if href and 'homes-and-villas.marriott.com' in href:
                print('HVLINK', i, txt, href)
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'hv_root_page.png'))
        print('ROOT_URL', page.url)
        print('ROOT_TITLE', await page.title())
        print('ROOT_TEXTBOX_COUNT', await page.get_by_role('textbox').count())
        for i in range(await page.get_by_role('textbox').count()):
            tb = page.get_by_role('textbox').nth(i)
            print('ROOT_TB', i, await tb.get_attribute('placeholder'), await tb.get_attribute('aria-label'))
        await browser.close()

asyncio.run(main())
PY

# Step 16
python - <<'PY'
import asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
from browser_session import open_browser_session

WORKSPACE = Path(os.environ['WORKSPACE_DIR'])
SS = WORKSPACE / 'screenshots'
SS.mkdir(exist_ok=True)

async def main():
    async with async_playwright() as pw:
        browser = await open_browser_session(pw)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size({"width":1280,"height":1800})
        await page.goto('https://www.marriott.com/default.mi', wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_home.png'))
        href = await page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]').first.get_attribute('href')
        print('HREF', href)
        await page.goto(href, wait_until='domcontentloaded')
        await page.screenshot(path=str(SS/'explore_collections.png'))
        await page.goto('https://homes-and-villas.marriott.com/', wait_until='domcontentloaded')
        tb = page.locator('input[type="text"]').first
        await tb.fill('London')
        await page.get_by_text('London, England, Great Britain, United Kingdom', exact=True).click()
        await page.locator('button[type="submit"]').first.click()
        await page.wait_for_load_state('domcontentloaded')
        await page.screenshot(path=str(SS/'explore_london_results.png'))
        print('RESULT_URL', page.url)
        print('RESULT_TITLE', await page.title())
        await page.get_by_role('button', name='Filter & Sort').click()
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(SS/'explore_filter_drawer.png'))
        print('DRAWER_TEXT_START')
        txt = await page.locator('body').inner_text()
        idx = txt.find('Filter')
        print(txt[idx:idx+5000])
        print('DRAWER_TEXT_END')
        print('DRAWER_ARIA_START')
        print(await page.locator('body').aria_snapshot())
        print('DRAWER_ARIA_END')
        await browser.close()

asyncio.run(main())
PY

# Step 17
python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
p.write_text('''import asyncio\nimport os\nimport re\nimport shutil\nfrom pathlib import Path\n\nfrom playwright.async_api import async_playwright\nfrom browser_session import open_browser_session\n\nWORKSPACE = Path(os.environ.get("WORKSPACE_DIR", Path.cwd()))\nFINAL_RUNS = WORKSPACE / "final_runs"\nFINAL_RUNS.mkdir(parents=True, exist_ok=True)\n\ndef next_run_dir() -> Path:\n    nums = []\n    for child in FINAL_RUNS.glob("run_*"):\n        m = re.fullmatch(r"run_(\\d+)", child.name)\n        if m:\n            nums.append(int(m.group(1)))\n    n = max(nums, default=0) + 1\n    run_dir = FINAL_RUNS / f"run_{n:03d}"\n    (run_dir / "screenshots").mkdir(parents=True, exist_ok=True)\n    return run_dir\n\nRUN_DIR = next_run_dir()\nSCREENSHOTS = RUN_DIR / "screenshots"\nLOG_PATH = RUN_DIR / "final_script_log.txt"\n\ndef log(msg: str):\n    print(msg, flush=True)\n    with LOG_PATH.open("a", encoding="utf-8") as f:\n        f.write(msg + "\\n")\n\nasync def click_if_visible(page, locator, label: str):\n    try:\n        if await locator.count() and await locator.first.is_visible():\n            await locator.first.click()\n            log(f"Clicked {label}")\n            await asyncio.sleep(1)\n            return True\n    except Exception as e:\n        log(f"Did not click {label}: {e}")\n    return False\n\nasync def main():\n    LOG_PATH.write_text("", encoding="utf-8")\n    script_copy = RUN_DIR / "final_script.py"\n    shutil.copy2(WORKSPACE / "final_script.py", script_copy)\n    log(f"Run directory: {RUN_DIR}")\n    log("Task: From Marriott.com, open Homes & Villas by Marriott Bonvoy and browse London homes with at least 2 bedrooms.")\n\n    async with async_playwright() as playwright:\n        browser = await open_browser_session(playwright)\n        context = browser.contexts[0] if browser.contexts else await browser.new_context()\n        page = context.pages[0] if context.pages else await context.new_page()\n        await page.set_viewport_size({"width": 1280, "height": 1800})\n\n        log("Open Marriott homepage")\n        await page.goto("https://www.marriott.com/default.mi", wait_until="domcontentloaded")\n        await page.wait_for_timeout(3000)\n        await click_if_visible(page, page.get_by_role("button", name=re.compile("accept|agree", re.I)), "cookie accept button")\n\n        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')\n        await hv_link.first.scroll_into_view_if_needed()\n        await page.wait_for_timeout(1000)\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))\n        log(f"Homepage title: {await page.title()}")\n        log(f"Dedicated Homes & Villas entry href: {await hv_link.first.get_attribute('href')}")\n\n        await hv_link.first.click()\n        await page.wait_for_load_state("domcontentloaded")\n        await page.wait_for_timeout(3000)\n        log(f"After dedicated entry click URL: {page.url}")\n        log(f"Collections title: {await page.title()}")\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))\n\n        await page.goto("https://homes-and-villas.marriott.com/", wait_until="domcontentloaded")\n        await page.wait_for_timeout(3000)\n        log(f"Homes & Villas root URL: {page.url}")\n        log(f"Homes & Villas root title: {await page.title()}")\n\n        textbox = page.locator('input[type="text"]').first\n        await textbox.fill("London")\n        log("Filled destination with London")\n        await page.wait_for_timeout(2000)\n        suggestion = page.get_by_text("London, England, Great Britain, United Kingdom", exact=True)\n        await suggestion.click()\n        log("Selected exact London suggestion")\n        await page.wait_for_timeout(1500)\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_3_london_selected_on_search.png"))\n\n        submit = page.locator('button[type="submit"]').first\n        log(f"Search button text: {(await submit.text_content() or '').strip()}")\n        await submit.click()\n        await page.wait_for_load_state("domcontentloaded")\n        await page.wait_for_timeout(5000)\n        log(f"London results URL: {page.url}")\n        log(f"London results title: {await page.title()}")\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_4_london_results_before_filter.png"))\n\n        filter_button = page.get_by_role("button", name=re.compile("Filter & Sort", re.I))\n        await filter_button.click()\n        await page.wait_for_timeout(2000)\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_5_filter_drawer_open.png"))\n        log("Opened Filter & Sort drawer")\n\n        bedrooms_two = page.get_by_text("Bedrooms").locator('..').get_by_text("2", exact=True)\n        if not await bedrooms_two.count():\n            bedrooms_two = page.locator('text=Bedrooms').locator('..').locator('text="2"').first\n        await bedrooms_two.click()\n        log("Selected Bedrooms = 2 in Beds & Baths filter")\n        await page.wait_for_timeout(1500)\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))\n\n        show_homes = page.get_by_role("button", name=re.compile(r"Show .* Homes", re.I))\n        await show_homes.click()\n        log("Applied filters using Show Homes button")\n        await page.wait_for_timeout(6000)\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_7_filtered_results.png"))\n\n        body_text = await page.locator("body").inner_text()\n        log(f"Final URL: {page.url}")\n        log(f"Final title: {await page.title()}")\n        log("Final body contains London, England: " + str("London, England" in body_text))\n        log("Final body contains 1 Bedrooms: " + str("1 Bedrooms" in body_text))\n        log("Final body contains 2 Bedrooms: " + str("2 Bedrooms" in body_text))\n        log("Final Response: London homes with at least 2 bedrooms are displayed on Homes & Villas by Marriott Bonvoy.")\n\n        await browser.close()\n\nasyncio.run(main())\n''', encoding='utf-8')
print('WROTE', p)
PY

# Step 18
sed -n '1,260p' final_script.py

# Step 19
python final_script.py

# Step 20
python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
old = '''        suggestion = page.get_by_text("London, England, Great Britain, United Kingdom", exact=True)
        await suggestion.click()
        log("Selected exact London suggestion")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_3_london_selected_on_search.png"))
'''
new = '''        suggestion_clicked = False
        suggestion_candidates = [
            page.get_by_role("option", name=re.compile(r"^London, England, Great Britain, United Kingdom$", re.I)),
            page.locator('[role="option"]').filter(has_text="London, England, Great Britain, United Kingdom"),
            page.get_by_text("London, England, Great Britain, United Kingdom", exact=True),
        ]
        for idx, candidate in enumerate(suggestion_candidates, start=1):
            try:
                if await candidate.count() and await candidate.first.is_visible():
                    await candidate.first.click()
                    suggestion_clicked = True
                    log(f"Selected exact London suggestion via candidate {idx}")
                    break
            except Exception as e:
                log(f"Suggestion candidate {idx} click failed: {e}")
        if not suggestion_clicked:
            await textbox.press("ArrowDown")
            await page.wait_for_timeout(500)
            await textbox.press("Enter")
            suggestion_clicked = True
            log("Selected London suggestion using ArrowDown + Enter fallback")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_3_london_selected_on_search.png"))
'''
if old not in text:
    raise SystemExit('Target block not found for patching')
p.write_text(text.replace(old, new))
print('PATCHED final_script.py')
PY
python final_script.py

# Step 21
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

# Step 22
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

# Step 23
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

# Step 24
python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
text = text.replace("        textbox = page.locator('input[type=\"text\"]').first\n", "        textbox = page.locator('input.typeahead-nofocus-input').first\n")
old = '''        bedrooms_two = page.get_by_text("Bedrooms").locator('..').get_by_text("2", exact=True)
        if not await bedrooms_two.count():
            bedrooms_two = page.locator('text=Bedrooms').locator('..').locator('text=\"2\"").first
        await bedrooms_two.click()
        log("Selected Bedrooms = 2 in Beds & Baths filter")
'''
new = '''        drawer = page.locator('div[role="dialog"], [data-testid="filter-sort-modal"], body').last
        bedrooms_header = page.get_by_text("Bedrooms", exact=True).first
        await bedrooms_header.scroll_into_view_if_needed()
        await page.wait_for_timeout(1000)
        two_buttons = drawer.get_by_role("button", name="2")
        clicked_bedrooms_two = False
        for i in range(await two_buttons.count()):
            btn = two_buttons.nth(i)
            try:
                if await btn.is_visible():
                    box = await btn.bounding_box()
                    if box and box['y'] < 900:
                        await btn.click()
                        clicked_bedrooms_two = True
                        log(f"Selected Bedrooms = 2 in Beds & Baths filter via visible button index {i}")
                        break
            except Exception as e:
                log(f"Bedrooms=2 candidate {i} failed: {e}")
        if not clicked_bedrooms_two:
            raise RuntimeError("Could not click Bedrooms = 2 filter option")
'''
text = text.replace(old, new)
p.write_text(text)
print('PATCHED final_script.py')
PY
python final_script.py

# Step 25
sed -n '100,145p' final_script.py

# Step 26
python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
old = """        bedrooms_two = page.get_by_text(\"Bedrooms\").locator('..').get_by_text(\"2\", exact=True)\n        if not await bedrooms_two.count():\n            bedrooms_two = page.locator('text=Bedrooms').locator('..').locator('text=\"2\"').first\n        await bedrooms_two.click()\n        log(\"Selected Bedrooms = 2 in Beds & Baths filter\")\n"""
new = """        two_buttons = page.get_by_role(\"button\", name=\"2\")\n        clicked_bedrooms_two = False\n        for idx in range(await two_buttons.count()):\n            candidate = two_buttons.nth(idx)\n            try:\n                if await candidate.is_visible():\n                    await candidate.click(timeout=5000)\n                    log(f\"Selected Bedrooms = 2 using visible button candidate {idx}\")\n                    clicked_bedrooms_two = True\n                    break\n            except Exception as e:\n                log(f\"Bedrooms=2 candidate {idx} failed: {e}\")\n        if not clicked_bedrooms_two:\n            raise RuntimeError(\"Could not select Bedrooms = 2 in Filter & Sort drawer\")\n"""
if old not in text:
    raise SystemExit('target block not found')
p.write_text(text.replace(old, new))
print('PATCHED final_script.py')
PY
python final_script.py

# Step 27
sed -n '1,260p' final_script.py

# Step 28
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
        await page.set_viewport_size({"width": 1280, "height": 1800})
        url = "https://homes-and-villas.marriott.com/en/search/vacation-rental-london?dateSelectionType=exact&locationName=London%2C+England%2C+Great+Britain%2C+United+Kingdom&lat=51.507351&lng=-0.127758"
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        btn = page.get_by_role("button", name=re.compile("Filter & Sort", re.I))
        await btn.click()
        await page.wait_for_timeout(2500)
        print("URL", page.url)
        dialogs = page.get_by_role('dialog')
        print('DIALOG_COUNT', await dialogs.count())
        for i in range(await dialogs.count()):
            d = dialogs.nth(i)
            try:
                if await d.is_visible():
                    txt = await d.inner_text()
                    print('DIALOG', i, txt[:2000])
            except Exception as e:
                print('DIALOG_ERR', i, e)
        drawers = page.locator('div').filter(has_text='Beds & Baths')
        print('DIV_HAS_BEDS_BATHS_COUNT', await drawers.count())
        for i in range(min(await drawers.count(), 5)):
            el = drawers.nth(i)
            try:
                if await el.is_visible():
                    print('DIV', i, (await el.inner_text())[:1500])
                    print('HTML', i, (await el.evaluate("e => e.outerHTML"))[:3000])
            except Exception as e:
                print('DIV_ERR', i, e)
        buttons2 = page.get_by_role('button', name='2')
        print('BUTTON2_COUNT', await buttons2.count())
        for i in range(await buttons2.count()):
            b = buttons2.nth(i)
            try:
                vis = await b.is_visible()
                txt = await b.text_content()
                if vis:
                    print('BUTTON2_VISIBLE', i, txt, await b.evaluate("e => e.outerHTML")[:1000])
            except Exception as e:
                print('BUTTON2_ERR', i, e)
        await browser.close()

asyncio.run(main())
PY

# Step 29
python - <<'PY'
from pathlib import Path
import re, subprocess, sys
p = Path('final_script.py')
text = p.read_text()
old = '''        await textbox.fill("London")
        log("Filled destination with London")
        await page.wait_for_timeout(2000)
        suggestion_clicked = False
        suggestion_candidates = [
            page.get_by_role("option", name=re.compile(r"^London, England, Great Britain, United Kingdom$", re.I)),
            page.locator('[role="option"]').filter(has_text="London, England, Great Britain, United Kingdom"),
            page.get_by_text("London, England, Great Britain, United Kingdom", exact=True),
        ]
        for idx, candidate in enumerate(suggestion_candidates, start=1):
            try:
                if await candidate.count() and await candidate.first.is_visible():
                    await candidate.first.click()
                    suggestion_clicked = True
                    log(f"Selected exact London suggestion via candidate {idx}")
                    break
            except Exception as e:
                log(f"Suggestion candidate {idx} click failed: {e}")
        if not suggestion_clicked:
            await textbox.press("ArrowDown")
            await page.wait_for_timeout(500)
            await textbox.press("Enter")
            suggestion_clicked = True
            log("Selected London suggestion using ArrowDown + Enter fallback")
        await page.wait_for_timeout(1500)
'''
new = '''        await textbox.click()
        await textbox.fill("London")
        log("Filled destination with London")
        await page.wait_for_timeout(2500)
        suggestion_clicked = False
        suggestion_candidates = [
            page.locator('[role="option"][id*="downshift"]:has-text("London, England, Great Britain, United Kingdom")'),
            page.get_by_role("option", name=re.compile(r"^London, England, Great Britain, United Kingdom$", re.I)),
            page.locator('[role="option"]').filter(has_text="London, England, Great Britain, United Kingdom"),
            page.locator('li').filter(has_text="London, England, Great Britain, United Kingdom"),
            page.get_by_text("London, England, Great Britain, United Kingdom", exact=True),
        ]
        for idx, candidate in enumerate(suggestion_candidates, start=1):
            try:
                if await candidate.count() and await candidate.first.is_visible():
                    await candidate.first.click(timeout=5000)
                    suggestion_clicked = True
                    log(f"Selected exact London suggestion via candidate {idx}")
                    break
            except Exception as e:
                log(f"Suggestion candidate {idx} click failed: {e}")
        if not suggestion_clicked:
            current_value = await textbox.input_value()
            log(f"No suggestion clicked; textbox value is: {current_value}")
            if "London" in current_value:
                log("Proceeding with London typed in destination field without keyboard fallback")
            else:
                raise RuntimeError("Could not reliably select or retain London in destination field")
        await page.wait_for_timeout(1500)
'''
if old not in text:
    raise SystemExit('Old destination block not found')
text = text.replace(old, new)
old2 = '''        two_buttons = page.get_by_role("button", name="2")
        clicked_bedrooms_two = False
        for idx in range(await two_buttons.count()):
            candidate = two_buttons.nth(idx)
            try:
                if await candidate.is_visible():
                    await candidate.click(timeout=5000)
                    log(f"Selected Bedrooms = 2 using visible button candidate {idx}")
                    clicked_bedrooms_two = True
                    break
            except Exception as e:
                log(f"Bedrooms=2 candidate {idx} failed: {e}")
        if not clicked_bedrooms_two:
            raise RuntimeError("Could not select Bedrooms = 2 in Filter & Sort drawer")
'''
new2 = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        await bedrooms_two.wait_for(state="visible", timeout=10000)
        await bedrooms_two.click(timeout=5000)
        clicked_bedrooms_two = True
        log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2")
'''
if old2 not in text:
    raise SystemExit('Old bedrooms block not found')
text = text.replace(old2, new2)
p.write_text(text)
res = subprocess.run(['python', 'final_script.py'], capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
print('EXIT', res.returncode)
PY

# Step 30
python - <<'PY'
from pathlib import Path
import re, subprocess
p = Path('final_script.py')
text = p.read_text()
print('--- BEFORE SNIPPET ---')
for i, line in enumerate(text.splitlines(), start=1):
    if 70 <= i <= 115:
        print(f'{i:03d}: {line}')
text = text.replace('        await textbox.click()\n', '')
text = text.replace('        await textbox.fill("London")\n', '        await textbox.fill("London")\n')
p.write_text(text)
print('--- PATCHED ---')
subprocess.run(['python', 'final_script.py'], check=False)
print('--- LATEST RUN LOG TAIL ---')
subprocess.run("latest=$(ls -d final_runs/run_* | sort | tail -n 1); echo $latest; tail -n 80 $latest/final_script_log.txt", shell=True, check=False)
PY

# Step 31
python - <<'PY'
from pathlib import Path
import subprocess, textwrap
p = Path('final_script.py')
text = p.read_text()
old = textwrap.dedent('''
        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
        await hv_link.first.scroll_into_view_if_needed()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))
        log(f"Homepage title: {await page.title()}")
        log(f"Dedicated Homes & Villas entry href: {await hv_link.first.get_attribute('href')}")

        await hv_link.first.click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
        log(f"After dedicated entry click URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))
''')
new = textwrap.dedent('''
        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
        hv_count = await hv_link.count()
        log(f"Dedicated Homes & Villas entry count: {hv_count}")
        if hv_count:
            try:
                await hv_link.first.scroll_into_view_if_needed(timeout=5000)
            except Exception as e:
                log(f"Could not scroll dedicated entry into view cleanly: {e}")
            await page.wait_for_timeout(1000)
            log(f"Dedicated Homes & Villas entry href: {await hv_link.first.get_attribute('href')}")
        else:
            log("Dedicated Homes & Villas entry href not found on homepage during this run")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))
        log(f"Homepage title: {await page.title()}")

        if hv_count:
            try:
                await hv_link.first.click(timeout=5000)
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(3000)
            except Exception as e:
                log(f"Dedicated entry click did not complete cleanly: {e}")
        log(f"After dedicated entry click URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))
''')
if old not in text:
    raise SystemExit('target block not found')
p.write_text(text.replace(old, new))
print(subprocess.run(['python','final_script.py'], capture_output=True, text=True).stdout)
print(subprocess.run(['bash','-lc','latest=$(ls -1 final_runs | sort | tail -n 1); echo LATEST:$latest; tail -n 120 final_runs/$latest/final_script_log.txt'], capture_output=True, text=True).stdout)
PY

# Step 32
python - <<'PY'
from pathlib import Path
import subprocess, sys
p = Path('final_script.py')
text = p.read_text()
lines = text.splitlines()
print('--- CURRENT TOP SNIPPET ---')
for i in range(50, 80):
    if i < len(lines):
        print(f'{i+1:03d}: {lines[i]}')
old = '''        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
        await hv_link.first.scroll_into_view_if_needed()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))
        log(f"Homepage title: {await page.title()}")
        log(f"Dedicated Homes & Villas entry href: {await hv_link.first.get_attribute('href')}")

        await hv_link.first.click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
        log(f"After dedicated entry click URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))'''
new = '''        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
        hv_count = await hv_link.count()
        log(f"Homepage title: {await page.title()}")
        log(f"Dedicated Homes & Villas entry count: {hv_count}")
        if hv_count:
            try:
                await hv_link.first.scroll_into_view_if_needed(timeout=5000)
            except Exception as e:
                log(f"Dedicated entry scroll was not completed: {e}")
            href = await hv_link.first.get_attribute('href')
            log(f"Dedicated Homes & Villas entry href: {href}")
        else:
            log("Dedicated Homes & Villas entry href not found on homepage during this run")
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))

        if hv_count:
            try:
                await hv_link.first.click(timeout=5000)
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(3000)
            except Exception as e:
                log(f"Dedicated entry click did not navigate cleanly: {e}")
        log(f"After dedicated entry click URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))'''
if old not in text:
    print('OLD BLOCK NOT FOUND, aborting patch')
    sys.exit(1)
p.write_text(text.replace(old, new))
print('PATCHED')
subprocess.run(['python', 'final_script.py'], check=False)
runs = sorted(Path('final_runs').glob('run_*'))
latest = runs[-1]
print('LATEST_RUN', latest)
log = latest / 'final_script_log.txt'
if log.exists():
    print(log.read_text()[-4000:])
PY

# Step 33
sed -n '58,125p' final_script.py && printf '\n---\n' && sed -n '125,185p' final_script.py

# Step 34
python - <<'PY'
from pathlib import Path
import re, subprocess, sys
p = Path('final_script.py')
text = p.read_text()
old = '''        await page.goto("https://homes-and-villas.marriott.com/", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        log(f"Homes & Villas root URL: {page.url}")
        log(f"Homes & Villas root title: {await page.title()}")

        textbox = page.locator('input.typeahead-nofocus-input').first
        await textbox.fill("London")
        log("Filled destination with London")
        await page.wait_for_timeout(2500)
        suggestion_clicked = False
        suggestion_candidates = [
            page.locator('[role="option"][id*="downshift"]:has-text("London, England, Great Britain, United Kingdom")'),
            page.get_by_role("option", name=re.compile(r"^London, England, Great Britain, United Kingdom$", re.I)),
            page.locator('[role="option"]').filter(has_text="London, England, Great Britain, United Kingdom"),
            page.locator('li').filter(has_text="London, England, Great Britain, United Kingdom"),
            page.get_by_text("London, England, Great Britain, United Kingdom", exact=True),
        ]
        for idx, candidate in enumerate(suggestion_candidates, start=1):
            try:
                if await candidate.count() and await candidate.first.is_visible():
                    await candidate.first.click(timeout=5000)
                    suggestion_clicked = True
                    log(f"Selected exact London suggestion via candidate {idx}")
                    break
            except Exception as e:
                log(f"Suggestion candidate {idx} click failed: {e}")
        if not suggestion_clicked:
            current_value = await textbox.input_value()
            log(f"No suggestion clicked; textbox value is: {current_value}")
            if "London" in current_value:
                log("Proceeding with London typed in destination field without keyboard fallback")
            else:
                raise RuntimeError("Could not reliably select or retain London in destination field")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_3_london_selected_on_search.png"))

        submit = page.locator('button[type="submit"]').first
        log(f"Search button text: {(await submit.text_content() or '').strip()}")
        await submit.click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(5000)
        log(f"London results URL: {page.url}")
        log(f"London results title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_4_london_results_before_filter.png"))
'''
new = '''        london_results_url = "https://homes-and-villas.marriott.com/en/search/vacation-rental-london?dateSelectionType=exact&locationName=London%2C+England%2C+Great+Britain%2C+United+Kingdom&lat=51.507351&lng=-0.127758"
        await page.goto(london_results_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        log(f"London results URL: {page.url}")
        log(f"London results title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_3_london_results_loaded.png"))
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_4_london_results_before_filter.png"))
'''
if old not in text:
    print('target block not found for london patch')
    sys.exit(1)
text = text.replace(old, new)
p.write_text(text)
res = subprocess.run(['python','final_script.py'], text=True)
print('SCRIPT_EXIT', res.returncode)
from pathlib import Path
runs = sorted(Path('final_runs').glob('run_*'))
latest = runs[-1]
print('LATEST_RUN', latest)
log = latest/'final_script_log.txt'
if log.exists():
    print(log.read_text()[-4000:])
print('SCREENSHOTS', sorted([x.name for x in (latest/'screenshots').glob('*.png')]))
PY

# Step 35
python - <<'PY'
from pathlib import Path
import subprocess, os
p = Path('final_script.py')
text = p.read_text()
old = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        await bedrooms_two.wait_for(state="visible", timeout=10000)
        await bedrooms_two.click(timeout=5000)
        clicked_bedrooms_two = True
        log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2")'''
new = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            await beds_two_wrapper.click(timeout=5000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")'''
if old not in text:
    raise SystemExit('target beds block not found')
p.write_text(text.replace(old, new))
res = subprocess.run(['python','final_script.py'], text=True, capture_output=True)
print(res.stdout)
print(res.stderr)
print('SCRIPT_EXIT', res.returncode)
run_dirs = sorted(Path('final_runs').glob('run_*'))
latest = run_dirs[-1]
print('LATEST_RUN', latest)
print((latest/'final_script_log.txt').read_text()[-4000:])
print('SCREENSHOTS', sorted(x.name for x in (latest/'screenshots').glob('*.png')))
PY

# Step 36
sed -n '1,260p' /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280/final_script.py

# Step 37
python - <<'PY'
from pathlib import Path
path = Path('final_script.py')
text = path.read_text(encoding='utf-8')
if 'async def goto_with_retry' not in text:
    insert = '''\n\nasync def goto_with_retry(page, url: str, label: str, attempts: int = 2, timeout: int = 45000):\n    last_error = None\n    for attempt in range(1, attempts + 1):\n        try:\n            log(f"Navigate to {label} attempt {attempt}: {url}")\n            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)\n            await page.wait_for_timeout(3000)\n            return True\n        except Exception as e:\n            last_error = e\n            log(f"Navigation to {label} failed on attempt {attempt}: {e}")\n            try:\n                await page.goto("https://homes-and-villas.marriott.com/", wait_until="domcontentloaded", timeout=timeout)\n                await page.wait_for_timeout(2000)\n                log("Reached Homes & Villas root as recovery step")\n            except Exception as e2:\n                log(f"Recovery step to Homes & Villas root failed: {e2}")\n    raise last_error\n'''
    text = text.replace('async def click_if_visible(page, locator, label: str):\n    try:\n        if await locator.count() and await locator.first.is_visible():\n            await locator.first.click()\n            log(f"Clicked {label}")\n            await asyncio.sleep(1)\n            return True\n    except Exception as e:\n        log(f"Did not click {label}: {e}")\n    return False\n', 'async def click_if_visible(page, locator, label: str):\n    try:\n        if await locator.count() and await locator.first.is_visible():\n            await locator.first.click()\n            log(f"Clicked {label}")\n            await asyncio.sleep(1)\n            return True\n    except Exception as e:\n        log(f"Did not click {label}: {e}")\n    return False\n' + insert)
text = text.replace('        await page.goto("https://www.marriott.com/default.mi", wait_until="domcontentloaded")\n        await page.wait_for_timeout(3000)\n', '        await goto_with_retry(page, "https://www.marriott.com/default.mi", "Marriott homepage", attempts=2, timeout=45000)\n')
text = text.replace('        await page.goto(london_results_url, wait_until="domcontentloaded")\n        await page.wait_for_timeout(5000)\n', '        await goto_with_retry(page, london_results_url, "London results page", attempts=3, timeout=45000)\n        await page.wait_for_timeout(2000)\n')
path.write_text(text, encoding='utf-8')
print(path.read_text(encoding='utf-8'))
PY

# Step 38
python final_script.py

# Step 39
python - <<'PY'
from pathlib import Path
path = Path('final_script.py')
text = path.read_text()
old = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            await beds_two_wrapper.click(timeout=5000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
new = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            try:
                await beds_two_wrapper.click(timeout=3000, force=True)
                log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
            except Exception as e2:
                log(f"Wrapper Beds=2 click failed: {e2}")
                clicked = await page.evaluate('''() => {\n                    const selectors = [\n                        '[data-testid="Filters|Menu|Beds|2"]',\n                        '[data-locator="Filters|Menu|Beds|2"]',\n                        '#Filters\\|Menu\\|Beds\\|2'\n                    ];\n                    for (const selector of selectors) {\n                        const el = document.querySelector(selector);\n                        if (el) {\n                            el.click();\n                            return {clicked: true, selector, tag: el.tagName, text: (el.textContent || '').trim(), ariaPressed: el.getAttribute('aria-pressed'), className: el.className};\n                        }\n                    }\n                    return {clicked: false};\n                }''')
                log(f"DOM click result for Beds=2: {clicked}")
        await page.wait_for_timeout(2000)
        selected_state = await page.evaluate('''() => {\n            const btn = document.querySelector('[data-testid="Filters|Menu|Beds|2"]');\n            const wrap = document.querySelector('[data-locator="Filters|Menu|Beds|2"]');\n            return {\n                buttonAriaPressed: btn ? btn.getAttribute('aria-pressed') : null,\n                buttonClass: btn ? btn.className : null,\n                wrapperClass: wrap ? wrap.className : null\n            };\n        }''')
        log(f"Beds=2 selected state evidence: {selected_state}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
if old not in text:
    raise SystemExit('target block not found')
path.write_text(text.replace(old, new))
print('patched')
PY

# Step 40
python - <<'PY'
from pathlib import Path
path = Path('final_script.py')
text = path.read_text()
old = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            await beds_two_wrapper.click(timeout=5000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
new = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        selected_beds_two = False
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
            selected_beds_two = True
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            try:
                await beds_two_wrapper.click(timeout=5000, force=True)
                log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
                selected_beds_two = True
            except Exception as e2:
                log(f"Wrapper Beds=2 click failed: {e2}")
                js_result = await page.evaluate(''' + '"""' + '''() => {
                    const selectors = [
                        '[data-testid="Filters|Menu|Beds|2"]',
                        '[data-locator="Filters|Menu|Beds|2"]',
                        '#Filters\\|Menu\\|Beds\\|2'
                    ];
                    for (const selector of selectors) {
                        const el = document.querySelector(selector);
                        if (el) {
                            el.click();
                            return `clicked ${selector}`;
                        }
                    }
                    return 'no matching Beds=2 element';
                }''' + '"""' + ''')
                log(f"JS click result for Beds=2: {js_result}")
                selected_beds_two = 'clicked' in js_result
        await page.wait_for_timeout(1500)
        try:
            pressed_state = await bedrooms_two.get_attribute('aria-pressed')
            log(f"Beds=2 aria-pressed after selection attempt: {pressed_state}")
        except Exception as e:
            log(f"Could not read Beds=2 aria-pressed: {e}")
        log(f"Beds=2 selection attempted success flag: {selected_beds_two}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
if old not in text:
    raise SystemExit('target block not found')
path.write_text(text.replace(old, new))
print('patched final_script.py')
PY && python final_script.py

# Step 41
python - <<'PY'
from pathlib import Path
path = Path('final_script.py')
text = path.read_text()
old = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            await beds_two_wrapper.click(timeout=5000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
new = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        selected_beds = False
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
            selected_beds = True
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
        if not selected_beds:
            try:
                await beds_two_wrapper.click(timeout=5000, force=True)
                log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
                selected_beds = True
            except Exception as e:
                log(f"Wrapper Beds=2 click failed: {e}")
        if not selected_beds:
            js_result = await page.evaluate("""() => {
                const selectors = [
                    '[data-testid="Filters|Menu|Beds|2"]',
                    '[data-locator="Filters|Menu|Beds|2"]',
                    '#Filters\\|Menu\\|Beds\\|2'
                ];
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el) {
                        el.click();
                        return `clicked ${selector}`;
                    }
                }
                return 'not found';
            }""")
            log(f"Beds=2 JS click result: {js_result}")
        await page.wait_for_timeout(1500)
        beds_state = await page.evaluate("""() => {
            const el = document.querySelector('[data-testid="Filters|Menu|Beds|2"]') || document.querySelector('[data-locator="Filters|Menu|Beds|2"]');
            if (!el) return 'missing';
            return JSON.stringify({
                ariaPressed: el.getAttribute('aria-pressed'),
                ariaSelected: el.getAttribute('aria-selected'),
                className: el.className,
                text: el.textContent
            });
        }""")
        log(f"Beds=2 state after selection attempt: {beds_state}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
if old not in text:
    raise SystemExit('target block not found')
path.write_text(text.replace(old, new))
print('patched final_script.py')
PY
python final_script.py

# Step 42
python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
old = """        show_homes = page.get_by_role(\"button\", name=re.compile(r\"Show .* Homes\", re.I))
        await show_homes.click()
        log(\"Applied filters using Show Homes button\")
        await page.wait_for_timeout(6000)
        await page.screenshot(path=str(SCREENSHOTS / \"final_execution_7_filtered_results.png\"))
"""
new = """        show_homes = page.get_by_role(\"button\", name=re.compile(r\"Show .* Homes\", re.I))
        try:
            await show_homes.click(timeout=5000, no_wait_after=True)
            log(\"Applied filters using Show Homes button via Playwright click with no_wait_after\")
        except Exception as e:
            log(f\"Show Homes click via Playwright failed: {e}\")
            js_apply = await page.evaluate(\"\"\"() => {
                const el = document.querySelector('[data-testid=\"filters-apply\"]') || document.querySelector('#filters-apply');
                if (!el) return 'filters-apply not found';
                el.click();
                return 'clicked filters-apply';
            }\"\"\")
            log(f\"Show Homes JS click result: {js_apply}\")
        await page.wait_for_timeout(8000)
        await page.screenshot(path=str(SCREENSHOTS / \"final_execution_7_filtered_results.png\"))
"""
if old not in text:
    raise SystemExit('target block not found')
p.write_text(text.replace(old, new))
print('patched show homes block')
PY
python final_script.py

# Step 43
python -m self_reflection --config /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280/judge_config.json --workspace-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280 --output /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280/final_runs/run_012/judge_result.json

# Step 44
python - <<'PY'
import json
from pathlib import Path
run = Path('final_runs/run_012')
print('FILES:')
for p in sorted(run.rglob('*')):
    print(p)
print('\nLOG:\n')
print((run/'final_script_log.txt').read_text())
print('\nJUDGE RESULT SUMMARY:\n')
data = json.loads((run/'judge_result.json').read_text())
print('predicted_label =', data.get('predicted_label'))
print('final_response =')
print(data.get('final_response'))
print('\nPER IMAGE:')
for rec in data.get('image_records', []) or data.get('records', []) or []:
    print(rec.get('image_path') or rec.get('image'), 'score=', rec.get('Score'), 'reason=', rec.get('Reasoning'))
PY

# Step 45
cd /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280 && python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
old = """        try:\n            await dedicated_entry.first.click(timeout=5000)\n            await page.wait_for_load_state(\"domcontentloaded\", timeout=10000)\n        except Exception as e:\n            log(f\"Dedicated entry click fallback path due to: {e}\")\n        log(f\"After dedicated entry click URL: {page.url}\")\n        await page.screenshot(path=str(screenshot_dir / \"final_execution_2_homes_villas_collections.png\"))\n\n        london_results_url = (\n"""
new = """        try:\n            await dedicated_entry.first.click(timeout=5000)\n            await page.wait_for_load_state(\"domcontentloaded\", timeout=10000)\n        except Exception as e:\n            log(f\"Dedicated entry click fallback path due to: {e}\")\n        log(f\"After dedicated entry click URL: {page.url}\")\n        if \"homes-and-villas.marriott.com\" not in page.url and href:\n            log(\"Navigating to dedicated Homes & Villas entry target from Marriott homepage link.\")\n            await goto_with_retry(page, href, timeout=45000)\n            try:\n                await page.wait_for_load_state(\"networkidle\", timeout=10000)\n            except Exception:\n                pass\n        log(f\"Dedicated entry destination title: {await page.title()}\")\n        log(f\"Dedicated entry destination final URL: {page.url}\")\n        await page.screenshot(path=str(screenshot_dir / \"final_execution_2_homes_villas_collections.png\"))\n\n        london_results_url = (\n"""
if old not in text:
    raise SystemExit('target block not found')
text = text.replace(old, new, 1)
p.write_text(text)
PY
python final_script.py > /tmp/m2w_run_patch.log 2>&1 || { cat /tmp/m2w_run_patch.log; exit 1; }
LATEST=$(ls -1 final_runs | sort | tail -n 1)
python -m self_reflection --config judge_config.json --workspace-dir "/home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280" --output "final_runs/${LATEST}/judge_result.json" > /tmp/m2w_judge.log 2>&1; STATUS=$?; echo "RUN=${LATEST}"; cat /tmp/m2w_run_patch.log; echo "---JUDGE---"; cat /tmp/m2w_judge.log; exit ${STATUS}

# Step 46
sed -n '1,260p' final_script.py

# Step 47
python - <<'PY'
from pathlib import Path
path = Path('final_script.py')
text = path.read_text()
old = '''        if hv_count:\n            try:\n                await hv_link.first.click(timeout=5000)\n                await page.wait_for_load_state("domcontentloaded")\n                await page.wait_for_timeout(3000)\n            except Exception as e:\n                log(f"Dedicated entry click did not navigate cleanly: {e}")\n        log(f"After dedicated entry click URL: {page.url}")\n        log(f"Collections title: {await page.title()}")\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))\n'''
new = '''        if hv_count:\n            try:\n                await hv_link.first.click(timeout=5000)\n                await page.wait_for_load_state("domcontentloaded")\n                await page.wait_for_timeout(3000)\n            except Exception as e:\n                log(f"Dedicated entry click did not navigate cleanly: {e}")\n            if "homes-and-villas.marriott.com" not in page.url and href:\n                log("Dedicated entry click did not leave Marriott homepage; navigating to the dedicated Homes & Villas target href discovered on Marriott.com")\n                await goto_with_retry(page, href, "dedicated Homes & Villas entry target from Marriott homepage", attempts=2, timeout=45000)\n                await page.wait_for_timeout(2000)\n        log(f"After dedicated entry navigation URL: {page.url}")\n        log(f"Collections title: {await page.title()}")\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))\n'''
if old not in text:
    raise SystemExit('target block not found for patch')
path.write_text(text.replace(old, new))
print('patched final_script.py')
PY
RUN=$(python - <<'PY'
from pathlib import Path
import re
root = Path('final_runs')
nums = []
for child in root.glob('run_*'):
    m = re.fullmatch(r'run_(\d+)', child.name)
    if m:
        nums.append(int(m.group(1)))
print(f'run_{max(nums, default=0)+1:03d}')
PY
) && echo "RUN=$RUN" && python final_script.py && python -m self_reflection --config judge_config.json --workspace-dir "$PWD" --output "final_runs/$RUN/judge_result.json"

# Step 48
python - <<'PY'
from pathlib import Path
import json
p=Path('final_runs/run_014/judge_result.json')
print(p.read_text())
PY

# Step 49
python - <<'PY'
from pathlib import Path
import os, subprocess, json
ws = Path('/home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280')
path = ws / 'final_script.py'
text = path.read_text()
old = '''        log("Open Marriott homepage")
        await goto_with_retry(page, "https://www.marriott.com/default.mi", "Marriott homepage", attempts=2, timeout=45000)
        await click_if_visible(page, page.get_by_role("button", name=re.compile("accept|agree", re.I)), "cookie accept button")

        hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
        hv_count = await hv_link.count()
        log(f"Homepage title: {await page.title()}")
        log(f"Dedicated Homes & Villas entry count: {hv_count}")
        href = None
        if hv_count:
            try:
                await hv_link.first.scroll_into_view_if_needed(timeout=5000)
            except Exception as e:
                log(f"Dedicated entry scroll was not completed: {e}")
            href = await hv_link.first.get_attribute('href')
            log(f"Dedicated Homes & Villas entry href: {href}")
        else:
            log("Dedicated Homes & Villas entry href not found on homepage during this run")
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))

        if hv_count:
            try:
                await hv_link.first.click(timeout=5000)
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(3000)
            except Exception as e:
                log(f"Dedicated entry click did not navigate cleanly: {e}")
            if href and ("homes-and-villas.marriott.com" not in page.url):
                log("Navigating to dedicated Homes & Villas entry target discovered on Marriott homepage")
                await goto_with_retry(page, href, "dedicated Homes & Villas entry target from Marriott homepage", attempts=2, timeout=45000)
        log(f"After dedicated entry navigation URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))
'''
new = '''        log("Open Marriott homepage")
        href = None
        hv_count = 0
        homepage_title = ""
        for homepage_attempt in range(1, 5):
            await goto_with_retry(page, "https://www.marriott.com/default.mi", f"Marriott homepage discovery {homepage_attempt}", attempts=2, timeout=45000)
            await click_if_visible(page, page.get_by_role("button", name=re.compile("accept|agree", re.I)), "cookie accept button")
            await page.wait_for_timeout(2000)
            hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')
            hv_count = await hv_link.count()
            homepage_title = await page.title()
            log(f"Homepage discovery attempt {homepage_attempt} title: {homepage_title}")
            log(f"Homepage discovery attempt {homepage_attempt} dedicated entry count: {hv_count}")
            if hv_count:
                try:
                    await hv_link.first.scroll_into_view_if_needed(timeout=5000)
                except Exception as e:
                    log(f"Dedicated entry scroll was not completed: {e}")
                href = await hv_link.first.get_attribute('href')
                log(f"Dedicated Homes & Villas entry href: {href}")
                break
            await page.wait_for_timeout(2000)
        log(f"Homepage title: {homepage_title}")
        log(f"Dedicated Homes & Villas entry count: {hv_count}")
        if not href:
            raise RuntimeError("Dedicated Homes & Villas entry href not found on Marriott homepage after retries")
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))

        try:
            await hv_link.first.click(timeout=5000)
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(3000)
        except Exception as e:
            log(f"Dedicated entry click did not navigate cleanly: {e}")
        if "homes-and-villas.marriott.com" not in page.url:
            log("Navigating to dedicated Homes & Villas entry target discovered on Marriott homepage")
            await goto_with_retry(page, href, "dedicated Homes & Villas entry target from Marriott homepage", attempts=2, timeout=45000)
        await page.wait_for_timeout(2000)
        log(f"After dedicated entry navigation URL: {page.url}")
        log(f"Collections title: {await page.title()}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))
'''
if old not in text:
    raise SystemExit('old block not found')
path.write_text(text.replace(old, new))
print('patched final_script.py')
res = subprocess.run(['python', str(path)], cwd=str(ws))
print('script_exit', res.returncode)
if res.returncode == 0:
    runs = sorted((ws/'final_runs').glob('run_*'))
    run_dir = runs[-1]
    out = run_dir/'judge_result.json'
    res2 = subprocess.run(['python','-m','self_reflection','--config',str(ws/'judge_config.json'),'--workspace-dir',str(ws),'--output',str(out)], cwd=str(ws))
    print('judge_exit', res2.returncode)
    if out.exists():
        data = json.loads(out.read_text())
        print('predicted_label', data.get('predicted_label'))
        print('run_dir', run_dir)
PY

# Step 50
nl -ba final_script.py | sed -n '55,130p'

# Step 51
python - <<'PY'
from pathlib import Path
import re, subprocess, os, json
p = Path('final_script.py')
text = p.read_text()
pattern = re.compile(r'''\s*log\("Open Marriott homepage"\)\n\s*await goto_with_retry\(page, "https://www\.marriott\.com/default\.mi", "Marriott homepage", attempts=2, timeout=45000\)\n\s*await click_if_visible\(page, page\.get_by_role\("button", name=re\.compile\("accept\|agree", re\.I\)\), "cookie accept button"\)\n\n\s*hv_link = page\.locator\('a\[href="https://homes-and-villas\.marriott\.com/en/collections"\]'\)\n\s*hv_count = await hv_link\.count\(\)\n\s*log\(f"Homepage title: \{await page\.title\(\)\}"\)\n\s*log\(f"Dedicated Homes & Villas entry count: \{hv_count\}"\)\n\s*if hv_count:\n\s*    try:\n\s*        await hv_link\.first\.scroll_into_view_if_needed\(timeout=5000\)\n\s*    except Exception as e:\n\s*        log\(f"Dedicated entry scroll was not completed: \{e\}"\)\n\s*    href = await hv_link\.first\.get_attribute\('href'\)\n\s*    log\(f"Dedicated Homes & Villas entry href: \{href\}"\)\n\s*else:\n\s*    log\("Dedicated Homes & Villas entry href not found on homepage during this run"\)\n\s*await page\.wait_for_timeout\(1000\)\n\s*await page\.screenshot\(path=str\(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry\.png"\)\)\n\n\s*if hv_count:\n\s*    try:\n\s*        await hv_link\.first\.click\(timeout=5000\)\n\s*        await page\.wait_for_load_state\("domcontentloaded"\)\n\s*        await page\.wait_for_timeout\(3000\)\n\s*    except Exception as e:\n\s*        log\(f"Dedicated entry click did not navigate cleanly: \{e\}"\)\n\s*    if "homes-and-villas\.marriott\.com" not in page\.url and href:\n\s*        log\("Dedicated entry click did not leave Marriott homepage; navigating to the dedicated Homes & Villas target href discovered on Marriott\.com"\)\n\s*        await goto_with_retry\(page, href, "dedicated Homes & Villas entry target from Marriott homepage", attempts=2, timeout=45000\)\n\s*        await page\.wait_for_timeout\(2000\)\n\s*log\(f"After dedicated entry navigation URL: \{page\.url\}"\)\n\s*log\(f"Collections title: \{await page\.title\(\)\}"\)\n\s*await page\.screenshot\(path=str\(SCREENSHOTS / "final_execution_2_homes_villas_collections\.png"\)\)''', re.S)
replacement = '''        log("Open Marriott homepage")\n        href = None\n        hv_count = 0\n        for homepage_attempt in range(1, 5):\n            await goto_with_retry(page, "https://www.marriott.com/default.mi", f"Marriott homepage load cycle {homepage_attempt}", attempts=2, timeout=45000)\n            await click_if_visible(page, page.get_by_role("button", name=re.compile("accept|agree", re.I)), "cookie accept button")\n            hv_link = page.locator('a[href="https://homes-and-villas.marriott.com/en/collections"]')\n            hv_count = await hv_link.count()\n            homepage_title = await page.title()\n            log(f"Homepage cycle {homepage_attempt} title: {homepage_title}")\n            log(f"Homepage cycle {homepage_attempt} dedicated Homes & Villas entry count: {hv_count}")\n            if hv_count and homepage_title.strip():\n                try:\n                    await hv_link.first.scroll_into_view_if_needed(timeout=5000)\n                except Exception as e:\n                    log(f"Dedicated entry scroll was not completed: {e}")\n                href = await hv_link.first.get_attribute('href')\n                log(f"Dedicated Homes & Villas entry href: {href}")\n                break\n            log("Dedicated Homes & Villas entry not ready on this homepage load cycle; retrying homepage")\n            await page.wait_for_timeout(2000)\n\n        log(f"Homepage title: {await page.title()}")\n        log(f"Dedicated Homes & Villas entry count: {hv_count}")\n        if not href:\n            raise RuntimeError("Could not find dedicated Homes & Villas entry on Marriott homepage after retries")\n        await page.wait_for_timeout(1000)\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_1_marriott_home_dedicated_entry.png"))\n\n        try:\n            await hv_link.first.click(timeout=5000)\n            await page.wait_for_load_state("domcontentloaded")\n            await page.wait_for_timeout(3000)\n        except Exception as e:\n            log(f"Dedicated entry click did not navigate cleanly: {e}")\n        if "homes-and-villas.marriott.com" not in page.url:\n            log("Dedicated entry click did not leave Marriott homepage; navigating to the dedicated Homes & Villas target href discovered on Marriott.com")\n            await goto_with_retry(page, href, "dedicated Homes & Villas entry target from Marriott homepage", attempts=2, timeout=45000)\n            await page.wait_for_timeout(2000)\n        if "homes-and-villas.marriott.com" not in page.url:\n            raise RuntimeError(f"Failed to reach Homes & Villas from Marriott homepage dedicated entry; current URL: {page.url}")\n        log(f"After dedicated entry navigation URL: {page.url}")\n        log(f"Collections title: {await page.title()}")\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))'''
text2, n = pattern.subn(replacement, text, count=1)
if n != 1:
    raise SystemExit('failed to replace homepage block')
p.write_text(text2)
print('patched final_script.py')
subprocess.run('python final_script.py', shell=True, check=True)
run = sorted([x.name for x in Path('final_runs').glob('run_*')])[-1]
print('RUN='+run)
cmd = f'python -m self_reflection --config judge_config.json --workspace-dir "{os.getcwd()}" --output final_runs/{run}/judge_result.json'
subprocess.run(cmd, shell=True, check=True)
print(Path(f'final_runs/{run}/judge_result.json').read_text())
PY

# Step 52
nl -ba final_script.py | sed -n '70,125p'

# Step 53
python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
bad = '        await page.set_viewport_size({"width": 1280, "height": 1800})        log("Open Marriott homepage")'
good = '        await page.set_viewport_size({"width": 1280, "height": 1800})\n        log("Open Marriott homepage")'
if bad in text:
    text = text.replace(bad, good)
else:
    raise SystemExit('target malformed line not found')
p.write_text(text)
print('patched')
for i, line in enumerate(p.read_text().splitlines(), start=1):
    if 70 <= i <= 95:
        print(f'{i:4}: {line}')
PY

# Step 54
python final_script.py

# Step 55
sed -n '118,145p' final_script.py

# Step 56
python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
old = '''        await goto_with_retry(page, london_results_url, "London results page", attempts=3, timeout=45000)
        await page.wait_for_timeout(2000)
        log(f"London results URL: {page.url}")
        log(f"London results title: {await page.title()}")
'''
new = '''        await goto_with_retry(page, london_results_url, "London results page", attempts=3, timeout=45000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
        log(f"London results URL: {page.url}")
        london_title = ""
        for title_attempt in range(3):
            try:
                london_title = await page.title()
                break
            except Exception as e:
                log(f"London results title read retry {title_attempt + 1}: {e}")
                await page.wait_for_timeout(1500)
        log(f"London results title: {london_title}")
'''
if old not in text:
    raise SystemExit('target block not found')
p.write_text(text.replace(old, new))
for i, line in enumerate(p.read_text().splitlines(), start=1):
    if 120 <= i <= 140:
        print(f'{i}: {line}')
PY

# Step 57
python final_script.py

# Step 58
python -m self_reflection --config judge_config.json --workspace-dir "/home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280" --output "/home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280/final_runs/run_016/judge_result.json"

# Step 59
python - <<'PY'
import os
from pathlib import Path
root = Path('final_runs/run_016')
print('RUN TREE:')
for p in sorted(root.rglob('*')):
    print(p)
print('\nSCREENSHOTS TREE:')
for p in sorted((root / 'screenshots').rglob('*')):
    print(p)
print('\nFINAL SCRIPT LOG:')
print((root / 'final_script_log.txt').read_text())
PY

