import os
import json
import re

def get_stats(run_path):
    tasks = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
    count_401_tasks = 0
    count_401_files = 0
    count_401_total = 0
    count_sr_and_401 = 0
    count_blocked = 0
    count_blocked_occurrences = 0
    reasons = {}
    
    limits_exceeded_tasks = []

    for task_id in tasks:
        task_dir = os.path.join(run_path, task_id)
        has_401 = False
        has_sr = False
        has_blocked = False
        blocked_count = 0
        task_401_files = 0
        
        files = []
        for root, _, filenames in os.walk(task_dir):
            for f in filenames:
                if f.endswith(('.sh', '.log', '.json', '.jsonl')):
                    files.append(os.path.join(root, f))
        
        for fpath in files:
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    content = f.read()
                    matches_401 = re.findall(r'401|Unauthorized|HTTPStatusError', content)
                    if matches_401:
                        has_401 = True
                        task_401_files += 1
                        count_401_total += len(matches_401)
                    if 'self_reflection' in content:
                        has_sr = True
                    matches_blocked = re.findall(r'Completion blocked', content)
                    if matches_blocked:
                        has_blocked = True
                        blocked_count += len(matches_blocked)
            except:
                pass
        
        if has_401:
            count_401_tasks += 1
            count_401_files += task_401_files
            if has_sr:
                count_sr_and_401 += 1
        
        if has_blocked:
            count_blocked += 1
            count_blocked_occurrences += blocked_count

        # Check judge_result for LimitsExceeded
        jr_path = None
        for root, _, filenames in os.walk(os.path.join(task_dir, 'final_runs')):
             if 'judge_result.json' in filenames:
                 jr_path = os.path.join(root, 'judge_result.json')
                 break
        
        exit_status = None
        if jr_path:
            try:
                with open(jr_path, 'r') as f:
                    jr = json.load(f)
                    exit_status = jr.get('exit_status')
            except:
                pass
        
        if exit_status == 'LimitsExceeded':
            limits_exceeded_tasks.append((task_id, has_blocked))

    return {
        'tasks_401': count_401_tasks,
        'files_401': count_401_files,
        'total_401': count_401_total,
        'sr_and_401': count_sr_and_401,
        'blocked_tasks': count_blocked,
        'blocked_total': count_blocked_occurrences,
        'limits_exceeded': limits_exceeded_tasks
    }

run_a = "/data/t-yifeili/mini-web-agent/outputs/best_default_judge_easy_20260623_171943"
run_b = "/data/t-yifeili/mini-web-agent/outputs/sft_state_debug_gpt54_phyagi/om2w_260220_easy_gpt54_step100_p32_local_20260623_201122"

stats_a = get_stats(run_a)
stats_b = get_stats(run_b)

print("Run A Stats:", stats_a)
print("Run B Stats:", stats_b)
