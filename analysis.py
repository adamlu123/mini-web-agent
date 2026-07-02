import os
import json
import re
import glob
import numpy as np
from collections import Counter, defaultdict

def categorize_command(cmd, step_num):
    cmd_lower = cmd.lower()
    if step_num == 1:
        return "init"
    if "cat > plan.md" in cmd or "cat > judge_config.json" in cmd:
        return "plan_judge_config_write"
    if "cat > final_script.py" in cmd:
        return "write_final_script"
    if "python final_script.py" in cmd or "python3 final_script.py" in cmd:
        return "run_final_script"
    if "playwright" in cmd_lower or "browser" in cmd_lower:
        return "explore_playwright"
    if "ls" in cmd_lower or "cat" in cmd_lower:
        if "final_runs" in cmd_lower or "run_" in cmd_lower:
            return "artifact_check_ls_cat"
        return "ls/cat"
    if "sed " in cmd_lower or "patch " in cmd_lower:
        return "small_patch_or_sed"
    if "env" in cmd_lower or "source " in cmd_lower or "export " in cmd_lower:
        return "env_or_source_inspect"
    if "reflect" in cmd_lower:
        return "self_reflection"
    return "other"

def analyze_run(run_path):
    task_dirs = sorted(glob.glob(os.path.join(run_path, "*/")))
    cmd_data = []
    for task_dir in task_dirs:
        history_path = os.path.join(task_dir, "command_history.sh")
        if not os.path.exists(history_path): continue
        with open(history_path, 'r') as f:
            content = f.read()
        steps = re.split(r'# Step \d+', content)
        for i, step_content in enumerate(steps):
            if not step_content.strip(): continue
            cmd = step_content.strip()
            cat = categorize_command(cmd, i)
            cmd_data.append({"cmd": cmd, "cat": cat, "len": len(cmd)})
    return cmd_data

def get_stats(lengths):
    if not lengths: return 0, 0, 0
    return np.mean(lengths), np.median(lengths), np.percentile(lengths, 90)

runA = "outputs/best_default_judge_easy_20260623_171943"
runB = "outputs/sft_state_debug_gpt54_phyagi/om2w_260220_easy_gpt54_step100_p32_local_20260623_180940"

dataA = analyze_run(runA)
dataB = analyze_run(runB)

print("### 1. Command Stats")
for name, data in [("Run A", dataA), ("Run B", dataB)]:
    lengths = [d['len'] for d in data]
    m, med, p90 = get_stats(lengths)
    print(f"{name}: Mean={m:.1f}, Med={med:.1f}, P90={p90:.1f}, Count={len(data)}")
    cats = defaultdict(list)
    for d in data: cats[d['cat']].append(d)
    for cat in sorted(cats.keys()):
        c_lens = [d['len'] for d in cats[cat]]
        print(f"  {cat:<25}: {len(c_lens):<4} | AvgLen: {np.mean(c_lens):.1f}")

print("\n### 2. Representative Commands (Run B Short selection)")
cats_B = defaultdict(list)
for d in dataB: cats_B[d['cat']].append(d)
for cat in sorted(cats_B.keys()):
    print(f"  [{cat}]")
    for d in cats_B[cat][:2]:
        print(f"    - {d['cmd'].replace('\n', ' ')[:100]}...")

print("\n### 3. B Gate Rejections")
rejection_cats = defaultdict(list)
for task_dir in sorted(glob.glob(os.path.join(runB, "*/"))):
    tid = os.path.basename(task_dir.rstrip('/'))
    gate_texts = []
    # Check trajectories and step observations
    paths = [os.path.join(task_dir, "trajectory.json")] + glob.glob(os.path.join(task_dir, "steps/step_*/observation.txt"))
    for p in paths:
        if not os.path.exists(p): continue
        try:
            with open(p, 'r') as f:
                content = f.read() if "json" not in p else json.dumps(json.load(f))
                for match in re.findall(r"(SelfReflectionGate.*?(?=\n\n|\Z)|Completion blocked.*?(?=\n\n|\Z))", content, re.S):
                    gate_texts.append(match)
        except: pass
    for text in set(gate_texts):
        reason = "other"
        if "no final_runs" in text or "final_runs/ does not exist" in text: reason = "no final_runs"
        elif "no run dirs" in text or "No run_ directories" in text: reason = "no run dirs"
        elif "judge_result.json' not found" in text or "judge_result missing" in text: reason = "judge_result missing"
        elif "predicted_label" in text and "not 1" in text: reason = "predicted_label not 1"
        elif "Failed to parse" in text or "Parse error" in text: reason = "parse error"
        elif "no screenshots" in text or "screenshots/ does not exist" in text: reason = "no screenshots"
        rejection_cats[reason].append({"tid": tid, "text": text.strip()})

for r, items in rejection_cats.items():
    print(f"  {r}: {len(items)}")
    for it in items[:3]:
        print(f"    - {it['tid']}: {it['text'][:120]}...")

print("\n### 4. self_reflect vs judge_result")
mismatch_tids = []
for task_dir in sorted(glob.glob(os.path.join(runB, "*/"))):
    tid = os.path.basename(task_dir.rstrip('/'))
    for run_dir in glob.glob(os.path.join(task_dir, "final_runs/run_*")):
        if os.path.exists(os.path.join(run_dir, "self_reflect_result.json")) and not os.path.exists(os.path.join(run_dir, "judge_result.json")):
            mismatch_tids.append(tid)
            break
print(f"  Mismatch Tasks: {len(mismatch_tids)}, Examples: {mismatch_tids[:5]}")

print("\n### 5. Workspace Path Check")
ws_counts = Counter()
for items in rejection_cats.values():
    for it in items:
        if "/workspace" in it['text']: ws_counts["/workspace"] += 1
        elif "outputs/" in it['text'] or "/data/t-yifeili" in it['text']: ws_counts["real_path"] += 1
        else: ws_counts["other"] += 1
print(f"  Path types: {dict(ws_counts)}")
