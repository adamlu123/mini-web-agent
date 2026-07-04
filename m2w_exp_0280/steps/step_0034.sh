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
