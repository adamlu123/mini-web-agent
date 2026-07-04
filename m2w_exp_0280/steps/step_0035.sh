python - <<'PY'
from pathlib import Path
import subprocess, os
p = Path('final_script.py')
text = p.read_text()
old = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        await bedrooms_two.wait_for(state="visible", timeout=10000)
        await bedrooms_two.click(timeout=5000)
        clicked_bedrooms_two = True
        log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2")'''
new = '''        bedrooms_two = page.locator('[data-testid="Filters|Menu|Beds|2"]').first
        beds_two_wrapper = page.locator('[data-locator="Filters|Menu|Beds|2"]').first
        try:
            await bedrooms_two.click(timeout=3000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter control Filters|Menu|Beds|2 with force click")
        except Exception as e:
            log(f"Direct Beds=2 button click failed: {e}")
            await beds_two_wrapper.click(timeout=5000, force=True)
            log("Selected Bedrooms = 2 using dedicated Beds filter wrapper Filters|Menu|Beds|2 with force click")'''
if old not in text:
    raise SystemExit('target beds block not found')
p.write_text(text.replace(old, new))
res = subprocess.run(['python','final_script.py'], text=True, capture_output=True)
print(res.stdout)
print(res.stderr)
print('SCRIPT_EXIT', res.returncode)
run_dirs = sorted(Path('final_runs').glob('run_*'))
latest = run_dirs[-1]
print('LATEST_RUN', latest)
print((latest/'final_script_log.txt').read_text()[-4000:])
print('SCREENSHOTS', sorted(x.name for x in (latest/'screenshots').glob('*.png')))
PY
