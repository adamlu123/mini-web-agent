python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
bad = '        await page.set_viewport_size({"width": 1280, "height": 1800})        log("Open Marriott homepage")'
good = '        await page.set_viewport_size({"width": 1280, "height": 1800})\n        log("Open Marriott homepage")'
if bad in text:
    text = text.replace(bad, good)
else:
    raise SystemExit('target malformed line not found')
p.write_text(text)
print('patched')
for i, line in enumerate(p.read_text().splitlines(), start=1):
    if 70 <= i <= 95:
        print(f'{i:4}: {line}')
PY
