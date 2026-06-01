"""Build a passed-task comparison report across eval-only runs.

Scans each run's ``exports/dumped_evals/eval_only/web_agent_om2w_easy_{train,val}.jsonl``
(written by ``echo_rl.web_agent.eval_entrypoint``), extracts the task intent +
starting URL from each rollout's prompt, and emits a markdown report listing
which model passed which task (pass@1, score>=1.0).

Usage:
    python -m echo_rl.web_agent.scripts.report_passed_tasks \
        --out eval_outputs/PASSED_TASKS.md

Edit RUNS below to point at the run dirs you want to compare.
"""

import argparse
import datetime
import json
import os
import re

# label -> run dir (under repo root). Edit to compare different runs.
RUNS = {
    "4B base":        ("Qwen3.5-4B", "base HF", "eval_outputs/4b_easy_20260529_011328"),
    "9B base":        ("Qwen3.5-9B", "base HF", "eval_outputs/9b_easy_20260529_020248"),
    "4B ckpt step_9": ("Qwen3.5-4B", "RL ckpt `global_step_9`", "eval_outputs/4b_easy_step9b_20260529_155041"),
}

TASK_RE = re.compile(r"Task:\s*(.*?)\n\nStarting URL:\s*(\S+)", re.S)


def load(run_dir):
    tasks = {}
    for split in ("train", "val"):
        f = os.path.join(run_dir, "exports", "dumped_evals", "eval_only",
                         f"web_agent_om2w_easy_{split}.jsonl")
        if not os.path.exists(f):
            continue
        for line in open(f):
            d = json.loads(line)
            m = TASK_RE.search(d["input_prompt"])
            intent = m.group(1).strip() if m else "<?>"
            url = m.group(2).strip() if m else "<?>"
            tasks[(split, intent, url)] = d["score"]
    return tasks


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_outputs/PASSED_TASKS.md")
    args = ap.parse_args()

    data = {name: load(rd) for name, (_, _, rd) in RUNS.items()}
    agg = {name: json.load(open(os.path.join(rd, "exports", "dumped_evals",
                                             "eval_only", "aggregated_results.jsonl")))
           for name, (_, _, rd) in RUNS.items()}

    def passed(name):
        return sum(1 for k in data[name] if data[name][k] >= 1.0)

    L = []
    L.append("# Online-Mind2Web (easy) — 통과 task 비교\n")
    L.append(f"_생성: {datetime.date.today()} · eval-only pass@1 (eval_n_samples_per_prompt=1) · "
             "o4-mini OSW judge · Browserbase Chromium_\n")
    L.append("80 task = train 68 + val 12. 평가 entrypoint: "
             "`echo_rl.web_agent.eval_entrypoint` (`colocate_all=false`).\n")

    L.append("## 1. 요약\n")
    L.append("| Model | Weights | all pass@1 (80) | train (68) | val (12) | avg turns | run dir |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, (mdl, w, rd) in RUNS.items():
        a = agg[name]
        L.append(f"| {mdl} | {w} | **{pct(a['eval/all/avg_score'])}** ({passed(name)}/80) | "
                 f"{pct(a['eval/web_agent_om2w_easy_train/avg_score'])} | "
                 f"{pct(a['eval/web_agent_om2w_easy_val/avg_score'])} | "
                 f"{a['eval/all/generate/avg_num_turns']:.1f} | `{os.path.basename(rd)}` |")

    passed_keys = sorted({k for name in data for k in data[name] if data[name][k] >= 1.0})
    L.append("\n## 2. 통과 task별 모델 비교 (어느 모델이 풀었나)\n")
    L.append("하나라도 통과한 task 합집합. ✅ = 통과, · = 실패/미통과.\n")
    L.append("| Split | Task query | Starting URL | 4B base | 9B base | 4B ckpt |")
    L.append("| --- | --- | --- | :---: | :---: | :---: |")
    for (split, intent, url) in passed_keys:
        def cell(name):
            v = data[name].get((split, intent, url))
            return "✅" if (v is not None and v >= 1.0) else "·"
        L.append(f"| {split} | {intent} | {url} | "
                 f"{cell('4B base')} | {cell('9B base')} | {cell('4B ckpt step_9')} |")

    L.append("\n## 3. 모델별 통과 task 전체 목록\n")
    for name, (mdl, w, _) in RUNS.items():
        pk = sorted(k for k in data[name] if data[name][k] >= 1.0)
        L.append(f"### {mdl} — {w}  ({len(pk)} passed)\n")
        for (split, intent, url) in pk:
            L.append(f"- **[{split}]** {intent}  — `{url}`")
        L.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
