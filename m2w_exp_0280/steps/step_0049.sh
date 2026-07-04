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
