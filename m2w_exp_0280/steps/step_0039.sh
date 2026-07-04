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
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            try:
                await beds_two_wrapper.click(timeout=3000, force=True)
                log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")
            except Exception as e2:
                log(f"Wrapper Beds=2 click failed: {e2}")
                clicked = await page.evaluate('''() => {\n                    const selectors = [\n                        '[data-testid="Filters|Menu|Beds|2"]',\n                        '[data-locator="Filters|Menu|Beds|2"]',\n                        '#Filters\\|Menu\\|Beds\\|2'\n                    ];\n                    for (const selector of selectors) {\n                        const el = document.querySelector(selector);\n                        if (el) {\n                            el.click();\n                            return {clicked: true, selector, tag: el.tagName, text: (el.textContent || '').trim(), ariaPressed: el.getAttribute('aria-pressed'), className: el.className};\n                        }\n                    }\n                    return {clicked: false};\n                }''')
                log(f"DOM click result for Beds=2: {clicked}")
        await page.wait_for_timeout(2000)
        selected_state = await page.evaluate('''() => {\n            const btn = document.querySelector('[data-testid="Filters|Menu|Beds|2"]');\n            const wrap = document.querySelector('[data-locator="Filters|Menu|Beds|2"]');\n            return {\n                buttonAriaPressed: btn ? btn.getAttribute('aria-pressed') : null,\n                buttonClass: btn ? btn.className : null,\n                wrapperClass: wrap ? wrap.className : null\n            };\n        }''')
        log(f"Beds=2 selected state evidence: {selected_state}")
        await page.screenshot(path=str(SCREENSHOTS / "final_execution_6_bedrooms_2_selected.png"))
'''
if old not in text:
    raise SystemExit('target block not found')
path.write_text(text.replace(old, new))
print('patched')
PY
