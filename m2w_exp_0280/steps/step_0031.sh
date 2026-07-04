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
