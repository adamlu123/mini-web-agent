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
