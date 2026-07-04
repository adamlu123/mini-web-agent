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
