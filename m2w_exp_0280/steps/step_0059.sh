python - <<'PY'
import os
from pathlib import Path
root = Path('final_runs/run_016')
print('RUN TREE:')
for p in sorted(root.rglob('*')):
    print(p)
print('\nSCREENSHOTS TREE:')
for p in sorted((root / 'screenshots').rglob('*')):
    print(p)
print('\nFINAL SCRIPT LOG:')
print((root / 'final_script_log.txt').read_text())
PY
