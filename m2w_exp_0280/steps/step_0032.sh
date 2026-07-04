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
