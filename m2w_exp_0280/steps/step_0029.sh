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
