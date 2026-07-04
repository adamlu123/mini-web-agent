python - <<'PY'
import json
from pathlib import Path
run = Path('final_runs/run_012')
print('FILES:')
for p in sorted(run.rglob('*')):
    print(p)
print('\nLOG:\n')
print((run/'final_script_log.txt').read_text())
print('\nJUDGE RESULT SUMMARY:\n')
data = json.loads((run/'judge_result.json').read_text())
print('predicted_label =', data.get('predicted_label'))
print('final_response =')
print(data.get('final_response'))
print('\nPER IMAGE:')
for rec in data.get('image_records', []) or data.get('records', []) or []:
    print(rec.get('image_path') or rec.get('image'), 'score=', rec.get('Score'), 'reason=', rec.get('Reasoning'))
PY
