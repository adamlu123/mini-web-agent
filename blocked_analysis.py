import os
import json
import re

run_b = "/data/t-yifeili/mini-web-agent/outputs/sft_state_debug_gpt54_phyagi/om2w_260220_easy_gpt54_step100_p32_local_20260623_201122"
tasks = [d for d in os.listdir(run_b) if os.path.isdir(os.path.join(run_b, d))]

reasons = {"judge_result missing": 0, "predicted_label not 1": 0, "other": 0}

for tid in tasks:
    tdir = os.path.join(run_b, tid)
    found_blocked = False
    for root, _, files in os.walk(tdir):
        for f in files:
            if f.endswith(('.log', '.json', '.jsonl')):
                try:
                    with open(os.path.join(root, f), 'r', errors='ignore') as f_in:
                        if 'Completion blocked' in f_in.read():
                            found_blocked = True
                            break
                except: pass
        if found_blocked: break
    
    if found_blocked:
        # classify
        jr_path = None
        for root, _, files in os.walk(os.path.join(tdir, 'final_runs')):
            if 'judge_result.json' in files:
                jr_path = os.path.join(root, 'judge_result.json')
                break
        
        if not jr_path:
            reasons["judge_result missing"] += 1
        else:
            try:
                with open(jr_path, 'r') as f:
                    jr = json.load(f)
                    if jr.get('predicted_label') != 1:
                        reasons["predicted_label not 1"] += 1
                    else:
                        reasons["other"] += 1
            except:
                reasons["other"] += 1

print("Run B Blocked Reasons:", reasons)
