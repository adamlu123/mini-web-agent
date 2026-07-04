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
