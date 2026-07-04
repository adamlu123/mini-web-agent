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
