import os
import json
import re

run_a = "/data/t-yifeili/mini-web-agent/outputs/best_default_judge_easy_20260623_171943"
run_b = "/data/t-yifeili/mini-web-agent/outputs/sft_state_debug_gpt54_phyagi/om2w_260220_easy_gpt54_step100_p32_local_20260623_201122"

def scan_run(run_path):
    results = {}
    if not os.path.exists(run_path):
        return results
    
    tasks = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
    for task_id in tasks:
        task_dir = os.path.join(run_path, task_id)
        task_data = {
            'has_401': False,
            '401_files': [],
            '401_count': 0,
            'has_self_reflection': False,
            'has_completion_blocked': False,
            'completion_blocked_count': 0,
            'completion_blocked_reason': None,
            'exit_status': None,
            'predicted_label': None,
            'step_count': 0,
            'endpoint': None,
            'context': []
        }
        
        # Check files
        files_to_check = ['command_history.sh', 'trajectory.json', 'runtime_errors.jsonl']
        logs_dir = os.path.join(task_dir, 'logs')
        if os.path.exists(logs_dir):
            files_to_check.extend([os.path.join('logs', f) for f in os.listdir(logs_dir) if f.endswith('.log')])
        
        final_runs_dir = os.path.join(task_dir, 'final_runs')
        if os.path.exists(final_runs_dir):
            for root, dirs, files in os.walk(final_runs_dir):
                if 'judge_result.json' in files:
                    files_to_check.append(os.path.relpath(os.path.join(root, 'judge_result.json'), task_dir))

        for rel_path in files_to_check:
            file_path = os.path.join(task_dir, rel_path)
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                    
                matches_401 = re.findall(r'401|Unauthorized|HTTPStatusError', content)
                if matches_401:
                    task_data['has_401'] = True
                    task_data['401_files'].append(rel_path)
                    task_data['401_count'] += len(matches_401)
                    # Extract context
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if any(x in line for x in ['401', 'Unauthorized', 'HTTPStatusError']):
                            # Hide keys/tokens
                            clean_line = re.sub(r'\"(Authorization|key|token)\":\s*\"[^\"]+\"', r'"\1": "HIDDEN"', line)
                            clean_line = re.sub(r'Bearer\s+[A-Za-z0-9\-\._~+/]+=*', 'Bearer HIDDEN', clean_line)
                            task_data['context'].append(f"{rel_path}:{idx+1}: {clean_line.strip()}"[:200])
                            if len(task_data['context']) >= 2:
                                break
                
                if 'self_reflection' in content:
                    task_data['has_self_reflection'] = True
                    if rel_path == 'command_history.sh':
                        match = re.search(r'--endpoint\s+([^\s]+)', content)
                        if match:
                            task_data['endpoint'] = match.group(1)

                matches_blocked = re.findall(r'Completion blocked', content)
                if matches_blocked:
                    task_data['has_completion_blocked'] = True
                    task_data['completion_blocked_count'] += len(matches_blocked)

            except Exception as e:
                pass

        # Summary info from trajectory.json or elsewhere
        traj_path = os.path.join(task_dir, 'trajectory.json')
        if os.path.exists(traj_path):
            try:
                with open(traj_path, 'r') as f:
                    traj = json.load(f)
                    task_data['step_count'] = len(traj)
                    if traj and isinstance(traj[-1], dict):
                        # Simple exit status heuristic
                        pass
            except:
                pass

        # Find status from judge_result.json
        for rel_path in files_to_check:
            if 'judge_result.json' in rel_path:
                try:
                    with open(os.path.join(task_dir, rel_path), 'r') as f:
                        judge = json.load(f)
                        task_data['exit_status'] = judge.get('exit_status')
                        task_data['predicted_label'] = judge.get('predicted_label')
                except:
                    pass

        # Classification for Completion blocked
        if task_data['has_completion_blocked']:
            # Search for judge_result.json
            found_jr = any('judge_result.json' in p for p in files_to_check if os.path.exists(os.path.join(task_dir, p)))
            if not found_jr:
                task_data['completion_blocked_reason'] = 'judge_result missing'
            elif task_data['predicted_label'] != 1:
                task_data['completion_blocked_reason'] = 'predicted_label not 1'
            else:
                # Check for "no screenshots" or "no final_runs"
                has_final_runs = os.path.exists(os.path.join(task_dir, 'final_runs'))
                if not has_final_runs:
                    task_data['completion_blocked_reason'] = 'no final_runs'
                else:
                    task_data['completion_blocked_reason'] = 'parse error/other'

        results[task_id] = task_data
    return results

data_a = scan_run(run_a)
data_b = scan_run(run_b)

def print_stats(name, data):
    tasks_401 = [t for t, d in data.items() if d['has_401']]
    files_401 = sum(len(d['401_files']) for d in data.values())
    total_401 = sum(d['401_count'] for d in data.values())
    sr_and_401 = [t for t, d in data.items() if d['has_401'] and d['has_self_reflection']]
    blocked = [t for t, d in data.items() if d['has_completion_blocked']]
    
    print(f"Run {name}:")
    print(f"  401/Unauthorized: {len(tasks_401)} tasks, {files_401} files, {total_401} total occurrences")
    print(f"  Both 401 and self_reflection: {len(sr_and_401)} tasks")
    print(f"  Completion blocked: {len(blocked)} tasks, {sum(data[t]['completion_blocked_count'] for t in blocked)} occurrences")
    
    reasons = {}
    for t in blocked:
        r = data[t]['completion_blocked_reason']
        reasons[r] = reasons.get(r, 0) + 1
    for r, c in reasons.items():
        print(f"    - {r}: {c} tasks")

print_stats("A (Best Default)", data_a)
print_stats("B (SFT State Debug)", data_b)

print("\n--- Details for Run B (401 tasks) ---")
for tid, d in data_b.items():
    if d['has_401']:
        print(f"Task: {tid}")
        print(f"  Exit Status: {d['exit_status']}, Predicted Label: {d['predicted_label']}, Steps: {d['step_count']}")
        print(f"  Files: {d['401_files']}")
        print(f"  Endpoint in self_reflection: {d['endpoint']}")
        print(f"  Context:")
        for ctx in d['context'][:2]:
            print(f"    {ctx}")

print("\n--- Analysis of Run A LimitsExceeded ---")
for tid, d in data_a.items():
    if d['exit_status'] == 'LimitsExceeded':
        print(f"Task {tid}: Completion blocked: {d['has_completion_blocked']}")
        if d['has_completion_blocked']:
             # Check if context contains reflection on failure or just judge missing
             reflection_related = any('self_reflection' in c.lower() for c in d['context'])
             print(f"  Reason: {d['completion_blocked_reason']}")

