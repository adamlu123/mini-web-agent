python - <<'PY'
from pathlib import Path
import json
ws = Path('/home/luyadong/sandbox/mini-web-agent/outputs/default/0601/N500_s100_agnostic_r2/m2w_exp_0280')
plan = ws/'plan.md'
plan.write_text('# Critical Points\n- [ ] CP1: Open Marriott.com and navigate to Homes & Villas by Marriott Bonvoy using the site\'s dedicated Homes & Villas entry point.\n- [ ] CP2: Apply the location/destination as London on the Homes & Villas experience so results are for London homes.\n- [ ] CP3: Apply the bedrooms filter using site controls so the search is limited to at least 2 bedrooms.\n- [ ] CP4: Display the filtered results page or listing set after applying the London and 2+ bedrooms constraints.\n')
cp_list = '1. CP1: Open Marriott.com and navigate to Homes & Villas by Marriott Bonvoy using the site\'s dedicated Homes & Villas entry point.\n2. CP2: Apply the location/destination as London on the Homes & Villas experience so results are for London homes.\n3. CP3: Apply the bedrooms filter using site controls so the search is limited to at least 2 bedrooms.\n4. CP4: Display the filtered results page or listing set after applying the London and 2+ bedrooms constraints.'
task_desc = 'From Marriott.com, open Homes & Villas by Marriott Bonvoy and browse London homes with at least 2 bedrooms.'
config = {
  'image_judge_system_prompt': 'You are a harsh evaluator of a single screenshot from a web task run. Return ONLY two labelled lines in exactly this format:\nReasoning: <1-2 sentences describing what the screenshot shows and which critical points it provides evidence for or against>\nScore: <integer 1-5, where 5 = this screenshot clearly evidences a critical point and 1 = this screenshot contains no relevant evidence>\nBe harsh when evidence is ambiguous, partially occluded, or does not clearly show the required state.',
  'image_judge_user_prompt': f'Task: {task_desc}\n\nCritical points to consider for this single image:\n{cp_list}\n\nEvaluate this screenshot against ALL critical points, even if it only evidences one of them. Be harsh when evidence is ambiguous or partially occluded. Return only the two labelled lines requested.',
  'final_verdict_system_prompt': 'You are a harsh aggregated judge of whether a web-task run succeeded. You will receive the task, critical points, the action log, per-image reasonings, and all screenshots. First write a Thoughts: block that evaluates every critical point one by one using the screenshots and log, noting any missing or ambiguous evidence. End your reply with EXACTLY one final line: Status: success or Status: failure',
  'final_verdict_user_prompt': f'Task: {task_desc}\n\nCritical points:\n{cp_list}\n\nAction history log:\n{{action_history_log}}\n\nPer-image reasonings:\n{{image_reasonings}}\n\nUsing the action log, per-image reasonings, and all attached screenshots, determine whether every critical point is satisfied. Be harsh: if a required filter, destination, or results state is not clearly evidenced, fail the run.'
}
(ws/'judge_config.json').write_text(json.dumps(config, indent=2))
print(plan.read_text())
print((ws/'judge_config.json').read_text())
PY
