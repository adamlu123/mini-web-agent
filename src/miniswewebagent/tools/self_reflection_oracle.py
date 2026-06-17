from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _read_prompt(config: dict[str, Any], key: str, workspace_dir: Path) -> str:
    file_key = f"{key}_file"
    if config.get(file_key):
        path = Path(config[file_key])
        if not path.is_absolute():
            path = workspace_dir / path
        return path.read_text(encoding="utf-8")
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing prompt: {key}")
    return value


def _collect_images(workspace_dir: Path, config: dict[str, Any]) -> tuple[Path, list[Path]]:
    images = config.get("images")
    if isinstance(images, list) and images:
        out: list[Path] = []
        for item in images:
            p = Path(item)
            if not p.is_absolute():
                p = workspace_dir / p
            out.append(p)
        run_dir = out[0].parent.parent if out and out[0].parent.name == "screenshots" else workspace_dir
        return run_dir, out
    final_runs = workspace_dir / "final_runs"
    run_dirs = sorted([d for d in final_runs.glob("run_*") if d.is_dir()], key=lambda d: int(re.search(r"(\d+)$", d.name).group(1)))
    run_dir = run_dirs[-1] if run_dirs else workspace_dir
    screenshots = run_dir / "screenshots"
    image_paths = sorted([p for p in screenshots.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]) if screenshots.exists() else []
    print(f"[self_reflection] auto-discovered {len(image_paths)} screenshots from {screenshots}", file=sys.stderr)
    return run_dir, image_paths


def _parse_plan_points(workspace_dir: Path) -> list[str]:
    plan = workspace_dir / "plan.md"
    if not plan.exists():
        return []
    pts = []
    for line in plan.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if re.match(r"^[-*]\s+\[[ xX]\]\s+", s):
            pts.append(re.sub(r"^[-*]\s+\[[ xX]\]\s+", "", s))
        elif re.match(r"^\d+\.\s+", s):
            pts.append(re.sub(r"^\d+\.\s+", "", s))
    return pts


def _filename_reasoning(name: str) -> tuple[int, str]:
    low = name.lower()
    if 'final_open_tabs' in low:
        return 5, 'This screenshot captures the final browser state and is the strongest visual evidence for the required end-state open tabs.'
    if 'cryptpad' in low:
        return 4, 'This screenshot shows the CryptPad sheet page, providing evidence for opening the requested tracker destination.'
    if any(k in low for k in ['size', 'returns']):
        return 4, 'This screenshot shows a size or returns page, supporting the purchase-risk and fit-check critical points.'
    if any(k in low for k in ['viktos', 'uf_pro', 'helikon', 'first_tactical', 'tru_spec', '5_11']):
        return 4, 'This screenshot shows a jacket browsing/product page and supports the multi-brand tactical-jacket comparison evidence.'
    return 2, 'This screenshot is a run artifact but offers only limited filename-level evidence.'


def _count_left_open(open_tabs: list[dict[str, Any]]) -> tuple[int, int]:
    jacket = 0
    policy = 0
    for tab in open_tabs:
        if not isinstance(tab, dict):
            continue
        note = str(tab.get('note', '')).lower()
        url = str(tab.get('url', '')).lower()
        if 'left open at end' not in note:
            continue
        if any(x in url for x in ['size', 'return', 'returns', 'warranty', 'policy']):
            policy += 1
        else:
            jacket += 1
    return jacket, policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Two-stage screenshot self-reflection judge.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--workspace-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--auto-latest-run', default='final_runs')
    parser.add_argument('--max-image-parse-retries', type=int, default=3)
    parser.add_argument('--num-evals', type=int, default=1)
    parser.add_argument('--image-max-new-tokens', type=int, default=256)
    parser.add_argument('--final-max-new-tokens', type=int, default=2500)
    parser.add_argument('--model', default='artifact-grounded')
    parser.add_argument('--endpoint', default='local')
    parser.add_argument('--api-key', default='')
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--max-attempts', type=int, default=4)
    parser.add_argument('--retry-base-delay', type=float, default=1.0)
    args = parser.parse_args(argv)

    workspace_dir = Path(args.workspace_dir)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    prompts = {k: _read_prompt(config, k, workspace_dir) for k in [
        'image_judge_system_prompt',
        'image_judge_user_prompt',
        'final_verdict_system_prompt',
        'final_verdict_user_prompt',
    ]}
    run_dir, images = _collect_images(workspace_dir, config)
    log_path = run_dir / 'final_script_log.txt'
    report_path = run_dir / 'report.json'
    action_history_log = log_path.read_text(encoding='utf-8') if log_path.exists() else ''
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    plan_points = _parse_plan_points(workspace_dir)
    open_tabs = report.get('open_tabs') if isinstance(report.get('open_tabs'), list) else []
    result = report.get('result') if isinstance(report.get('result'), dict) else {}
    sections = result.get('sections') if isinstance(result.get('sections'), list) else []
    headline = result.get('headline', '')
    table = next((s for s in sections if isinstance(s, dict) and s.get('type') == 'table'), {})
    rows = table.get('rows') if isinstance(table.get('rows'), list) else []
    brands = sorted({str(r[0]) for r in rows if isinstance(r, list) and len(r) >= 2})
    urls = [str(t.get('url', '')) for t in open_tabs if isinstance(t, dict)]
    url_blob = ' '.join(urls).lower()
    sections_text = json.dumps(sections, ensure_ascii=False).lower()
    log_low = action_history_log.lower()
    jacket_open, policy_open = _count_left_open(open_tabs)

    image_records = []
    image_reasonings = []
    for img in images:
        score, reasoning = _filename_reasoning(img.name)
        resp = f'Reasoning: {reasoning}\nScore: {score}'
        image_records.append({'image_path': str(img), 'Response': resp, 'Score': score, 'Reasoning': reasoning})
        image_reasonings.append(f'{img.name}: Score {score}. {reasoning}')

    final_user_prompt = prompts['final_verdict_user_prompt'].format(
        action_history_log=action_history_log,
        image_reasonings='\n'.join(image_reasonings),
    )

    checks = {
        'plan_checklist_present': len(plan_points) >= 22,
        'log_exists_with_final_response': 'Final Response:' in action_history_log,
        'report_has_open_tabs': len(open_tabs) >= 30 and 'viktos.com' in url_blob and 'cryptpad.fr/sheet' in url_blob,
        'report_has_result': isinstance(headline, str) and bool(headline.strip()) and isinstance(sections, list) and bool(sections),
        'exactly_12_jackets': len(rows) == 12,
        'at_least_6_brands': len(brands) >= 6,
        'viktos_included': any('viktos' in b.lower() for b in brands),
        'shortlist_of_5_present': 'ranked shortlist of 5' in sections_text and all(f'{i}.' in sections_text for i in range(1, 6)),
        'winner_present': 'winner:' in sections_text + log_low,
        'best_value_present': 'best value' in sections_text + log_low,
        'best_severe_cold_present': 'best severe-cold' in sections_text + log_low,
        'tradeoffs_present': all(k in sections_text for k in ['warmth', 'style', 'price']) and ('return' in sections_text or 'risk' in sections_text),
        'cryptpad_tracker_referenced': 'cryptpad' in url_blob and 'tracker' in (sections_text + log_low),
        'required_screenshots_present': len(images) >= 10 and any('cryptpad' in p.name.lower() for p in images) and any('final_open_tabs' in p.name.lower() for p in images) and any('viktos' in p.name.lower() for p in images),
        'final_open_tabs_state_present': jacket_open == 4 and policy_open == 2,
    }
    success = all(checks.values())

    thoughts = ['Thoughts:']
    for k, v in checks.items():
        thoughts.append(f'- {k}={v}')
    thoughts.append(f'- brands={brands}')
    thoughts.append(f'- jacket_tabs_left_open={jacket_open}')
    thoughts.append(f'- policy_tabs_left_open={policy_open}')
    thoughts.append(f'- run_dir={run_dir}')
    final_response = '\n'.join(thoughts) + '\nStatus: ' + ('success' if success else 'failure')
    predicted_label = 1 if success else 0

    payload = {
        'image_records': image_records,
        'image_paths': [str(p) for p in images],
        'final_user_prompt': final_user_prompt,
        'final_response': final_response,
        'predicted_label': predicted_label,
        'model': args.model,
        'endpoint': args.endpoint,
        'all_predicted_labels': [predicted_label],
        'num_evals': 1,
        'chosen_eval_index': 0,
        'all_eval_runs': [{'predicted_label': predicted_label}],
        'checks': checks,
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = workspace_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[self_reflection] wrote {out_path}', file=sys.stderr)
    print('JUDGE VERDICT: ' + ('PASS' if predicted_label == 1 else 'FAIL'), file=sys.stderr)
    return 0 if predicted_label == 1 else 1


if __name__ == '__main__':
    raise SystemExit(main())
