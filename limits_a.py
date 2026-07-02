import os
import json
import re

run_a = "/data/t-yifeili/mini-web-agent/outputs/best_default_judge_easy_20260623_171943"
tasks = [d for d in os.listdir(run_a) if os.path.isdir(os.path.join(run_a, d))]

print("Run A LimitsExceeded Tasks Analysis:")
for tid in tasks:
    tdir = os.path.join(run_a, tid)
    jr_path = None
    for root, _, files in os.walk(os.path.join(tdir, 'final_runs')):
        if 'judge_result.json' in files:
            jr_path = os.path.join(root, 'judge_result.json')
            break
    
    if jr_path:
        with open(jr_path, 'r') as f:
            jr = json.load(f)
            if jr.get('exit_status') == 'LimitsExceeded':
                # Check for Completion blocked
                has_blocked = False
                blocked_context = ""
                for root, _, files in os.walk(tdir):
                    for f in files:
                        if f.endswith(('.log', '.json', '.jsonl')):
                            try:
                                with open(os.path.join(root, f), 'r', errors='ignore') as f_in:
                                    content = f_in.read()
                                    if 'Completion blocked' in content:
                                        has_blocked = True
                                        # Get a bit of context
                                        match = re.search(r'.{0,100}Completion blocked.{0,100}', content)
                                        if match:
                                            blocked_context = match.group(0).replace('\n', ' ')
                                        break
                            except: pass
                    if has_blocked: break
                print(f"Task {tid}: Blocked={has_blocked}, Context={blocked_context[:100]}")

