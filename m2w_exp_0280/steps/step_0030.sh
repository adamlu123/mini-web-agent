python - <<'PY'
from pathlib import Path
import re, subprocess
p = Path('final_script.py')
text = p.read_text()
print('--- BEFORE SNIPPET ---')
for i, line in enumerate(text.splitlines(), start=1):
    if 70 <= i <= 115:
        print(f'{i:03d}: {line}')
text = text.replace('        await textbox.click()\n', '')
text = text.replace('        await textbox.fill("London")\n', '        await textbox.fill("London")\n')
p.write_text(text)
print('--- PATCHED ---')
subprocess.run(['python', 'final_script.py'], check=False)
print('--- LATEST RUN LOG TAIL ---')
subprocess.run("latest=$(ls -d final_runs/run_* | sort | tail -n 1); echo $latest; tail -n 80 $latest/final_script_log.txt", shell=True, check=False)
PY
