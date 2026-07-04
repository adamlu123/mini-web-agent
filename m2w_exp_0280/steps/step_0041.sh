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
        selected_beds = False
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
            selected_beds = True
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
        if not selected_beds:
            try:
                await beds_two_wrapper.click(timeout=5000, force=True)
                log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
                selected_beds = True
            except Exception as e:
                log(f"Wrapper Beds=2 click failed: {e}")
        if not selected_beds:
            js_result = await page.evaluate("""() => {
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
                return 'not found';
            }""")
            log(f"Beds=2 JS click result: {js_result}")
        await page.wait_for_timeout(1500)
        beds_state = await page.evaluate("""() => {
            const el = document.querySelector('[data-testid="Filters|Menu|Beds|2"]') || document.querySelector('[data-locator="Filters|Menu|Beds|2"]');
            if (!el) return 'missing';
            return JSON.stringify({
                ariaPressed: el.getAttribute('aria-pressed'),
                ariaSelected: el.getAttribute('aria-selected'),
                className: el.className,
                text: el.textContent
            });
        }""")
        log(f"Beds=2 state after selection attempt: {beds_state}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
if old not in text:
    raise SystemExit('target block not found')
path.write_text(text.replace(old, new))
print('patched final_script.py')
PY
python final_script.py
