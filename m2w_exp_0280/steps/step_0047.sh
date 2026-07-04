python - <<'PY'
from pathlib import Path
path = Path('final_script.py')
text = path.read_text()
old = '''        if hv_count:\n            try:\n                await hv_link.first.click(timeout=5000)\n                await page.wait_for_load_state("domcontentloaded")\n                await page.wait_for_timeout(3000)\n            except Exception as e:\n                log(f"Dedicated entry click did not navigate cleanly: {e}")\n        log(f"After dedicated entry click URL: {page.url}")\n        log(f"Collections title: {await page.title()}")\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))\n'''
new = '''        if hv_count:\n            try:\n                await hv_link.first.click(timeout=5000)\n                await page.wait_for_load_state("domcontentloaded")\n                await page.wait_for_timeout(3000)\n            except Exception as e:\n                log(f"Dedicated entry click did not navigate cleanly: {e}")\n            if "homes-and-villas.marriott.com" not in page.url and href:\n                log("Dedicated entry click did not leave Marriott homepage; navigating to the dedicated Homes & Villas target href discovered on Marriott.com")\n                await goto_with_retry(page, href, "dedicated Homes & Villas entry target from Marriott homepage", attempts=2, timeout=45000)\n                await page.wait_for_timeout(2000)\n        log(f"After dedicated entry navigation URL: {page.url}")\n        log(f"Collections title: {await page.title()}")\n        await page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homes_villas_collections.png"))\n'''
if old not in text:
    raise SystemExit('target block not found for patch')
path.write_text(text.replace(old, new))
print('patched final_script.py')
PY
RUN=$(python - <<'PY'
from pathlib import Path
import re
root = Path('final_runs')
nums = []
for child in root.glob('run_*'):
    m = re.fullmatch(r'run_(\d+)', child.name)
    if m:
        nums.append(int(m.group(1)))
print(f'run_{max(nums, default=0)+1:03d}')
PY
) && echo "RUN=$RUN" && python final_script.py && python -m self_reflection --config judge_config.json --workspace-dir "$PWD" --output "final_runs/$RUN/judge_result.json"
