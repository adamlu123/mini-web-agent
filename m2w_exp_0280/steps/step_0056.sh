python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
old = '''        await goto_with_retry(page, london_results_url, "London results page", attempts=3, timeout=45000)
        await page.wait_for_timeout(2000)
        log(f"London results URL: {page.url}")
        log(f"London results title: {await page.title()}")
'''
new = '''        await goto_with_retry(page, london_results_url, "London results page", attempts=3, timeout=45000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
        log(f"London results URL: {page.url}")
        london_title = ""
        for title_attempt in range(3):
            try:
                london_title = await page.title()
                break
            except Exception as e:
                log(f"London results title read retry {title_attempt + 1}: {e}")
                await page.wait_for_timeout(1500)
        log(f"London results title: {london_title}")
'''
if old not in text:
    raise SystemExit('target block not found')
p.write_text(text.replace(old, new))
for i, line in enumerate(p.read_text().splitlines(), start=1):
    if 120 <= i <= 140:
        print(f'{i}: {line}')
PY
