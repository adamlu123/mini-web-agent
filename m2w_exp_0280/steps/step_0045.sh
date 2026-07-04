cd /home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280 && python - <<'PY'
from pathlib import Path
p = Path('final_script.py')
text = p.read_text()
old = """        try:\n            await dedicated_entry.first.click(timeout=5000)\n            await page.wait_for_load_state(\"domcontentloaded\", timeout=10000)\n        except Exception as e:\n            log(f\"Dedicated entry click fallback path due to: {e}\")\n        log(f\"After dedicated entry click URL: {page.url}\")\n        await page.screenshot(path=str(screenshot_dir / \"final_execution_2_homes_villas_collections.png\"))\n\n        london_results_url = (\n"""
new = """        try:\n            await dedicated_entry.first.click(timeout=5000)\n            await page.wait_for_load_state(\"domcontentloaded\", timeout=10000)\n        except Exception as e:\n            log(f\"Dedicated entry click fallback path due to: {e}\")\n        log(f\"After dedicated entry click URL: {page.url}\")\n        if \"homes-and-villas.marriott.com\" not in page.url and href:\n            log(\"Navigating to dedicated Homes & Villas entry target from Marriott homepage link.\")\n            await goto_with_retry(page, href, timeout=45000)\n            try:\n                await page.wait_for_load_state(\"networkidle\", timeout=10000)\n            except Exception:\n                pass\n        log(f\"Dedicated entry destination title: {await page.title()}\")\n        log(f\"Dedicated entry destination final URL: {page.url}\")\n        await page.screenshot(path=str(screenshot_dir / \"final_execution_2_homes_villas_collections.png\"))\n\n        london_results_url = (\n"""
if old not in text:
    raise SystemExit('target block not found')
text = text.replace(old, new, 1)
p.write_text(text)
PY
python final_script.py > /tmp/m2w_run_patch.log 2>&1 || { cat /tmp/m2w_run_patch.log; exit 1; }
LATEST=$(ls -1 final_runs | sort | tail -n 1)
python -m self_reflection --config judge_config.json --workspace-dir "/home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280" --output "final_runs/${LATEST}/judge_result.json" > /tmp/m2w_judge.log 2>&1; STATUS=$?; echo "RUN=${LATEST}"; cat /tmp/m2w_run_patch.log; echo "---JUDGE---"; cat /tmp/m2w_judge.log; exit ${STATUS}
