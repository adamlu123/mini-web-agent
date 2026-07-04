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
