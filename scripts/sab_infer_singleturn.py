#!/usr/bin/env python3
"""Single-turn ScienceAgentBench inference for a science-SFT checkpoint.

One chat completion per task (system + task -> one response), then extract the
program and write it to `pred_programs/pred_<gold>.py` for the SAB harness. This
replaces the multi-turn agent loop for models trained on the single-turn D3
science SFT data (make_d3gym_science_sft.py).

Supports BOTH assistant output formats (that is the "two modes"):
  * sft_state XML : <think>...</think><bash>PROGRAM</bash><done>true</done>
  * code fence    : ```python\nPROGRAM\n```
Extraction prefers <bash>, then a ```python``` fence, then any fence, then the
raw text. A <bash> body that itself wraps a ```python``` fence is unwrapped.

The system prompt defaults to make_d3gym_science_sft.SYSTEM_PROMPT (so eval
matches training); override with --system-prompt-file.

Usage:
  python scripts/sab_infer_singleturn.py \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
    --model d3gym_9b \
    --tasks-file src/miniswewebagent/run/benchmarks/sab_verified.json \
    --pred-out /data/t-yifeili/ScienceAgentBench/pred_programs_d3gym_9b \
    --run-log  /data/t-yifeili/ScienceAgentBench/sab_d3gym_9b_run.jsonl
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

_REPO = Path(__file__).resolve().parent.parent
_DATA_SCRIPTS = _REPO / "LlamaFactory" / "scripts"


def default_system_prompt() -> str:
    sys.path.insert(0, str(_DATA_SCRIPTS))
    from make_d3gym_science_sft import SYSTEM_PROMPT  # noqa: E402

    return SYSTEM_PROMPT


def _unwrap_heredoc(s: str) -> str | None:
    """If s runs a program via `python[3] - <<'DELIM' ...`, return the body. The
    closing DELIM line is optional — base models sometimes hit max_tokens before
    closing it, so we take everything up to the closer if present, else to the end
    (dropping trailing </bash>/<done>/<final_response> tags)."""
    m = re.search(r"python3?\s+-\s*<<-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?[^\n]*\n", s)
    if not m:
        return None
    delim = m.group(1)
    rest = s[m.end():]
    cm = re.search(rf"(?m)^{re.escape(delim)}[ \t]*$", rest)
    body = rest[: cm.start()] if cm else rest
    return re.sub(r"\s*</?(bash|done|final_response)>.*$", "", body, flags=re.S)


def extract_program(text: str) -> str:
    """Return the program from any of: a <bash> heredoc (`python - <<'PY' ... PY`,
    closing tags optional), a raw <bash> body, or a ```python``` fence."""
    # <bash> block — tolerate a missing </bash> (match to end of text).
    m = re.search(r"<bash>\s*(.*?)(?:</bash>|\Z)", text, re.S)
    bash = m.group(1).strip() if (m and m.group(1).strip()) else None
    if bash is not None:
        body = _unwrap_heredoc(bash)
        if body is not None:
            return body.strip("\n")
        fm = re.search(r"```(?:python)?\s*\n?(.*?)```", bash, re.S)  # nested fence
        if fm:
            return fm.group(1).strip("\n")
        return re.sub(r"\s*</?(bash|done|final_response)>.*$", "", bash, flags=re.S).strip("\n")
    body = _unwrap_heredoc(text)               # heredoc without <bash> tags
    if body is not None:
        return body.strip("\n")
    fences = re.findall(r"```python\s*\n(.*?)```", text, re.S)  # closed python fence
    if fences:
        return fences[-1].strip("\n")
    m = re.search(r"```python\s*\n(.*)\Z", text, re.S)          # unclosed python fence
    if m and m.group(1).strip():
        return m.group(1).strip("\n")
    fences = re.findall(r"```\s*\n?(.*?)```", text, re.S)       # closed any fence
    if fences:
        return fences[-1].strip("\n")
    m = re.search(r"```\w*\s*\n(.*)\Z", text, re.S)             # unclosed any fence
    if m and m.group(1).strip():
        return m.group(1).strip("\n")
    return text.strip()  # last resort


def task_messages(task: dict, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["confirmed_task"]},
    ]


def one_task(task: dict, endpoint: str, model: str, system_prompt: str,
             max_tokens: int, timeout: float, temperature: float = 0.0) -> tuple[str, str, bool, str]:
    payload = {
        "model": model,
        "messages": task_messages(task, system_prompt),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    gold = task["sab"]["gold_program_name"]
    try:
        r = httpx.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        prog = extract_program(text)
        if not prog.strip():
            return gold, "ERROR", False, text
        return gold, prog, True, text
    except Exception as exc:  # noqa: BLE001
        return gold, f"ERROR\n# inference failed: {exc}", False, f"[EXC] {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--tasks-file",
        default=str(_REPO / "src/miniswewebagent/run/benchmarks/sab_verified.json"),
    )
    ap.add_argument("--pred-out", required=True)
    ap.add_argument("--run-log", default="")
    ap.add_argument("--raw-dir", default="", help="Dump each task's raw model response here.")
    ap.add_argument("--resume", action="store_true", help="Reuse completed responses in --raw-dir.")
    ap.add_argument("--system-prompt-file", default="", help="Override the default system prompt.")
    ap.add_argument("--task-ids", nargs="*", default=[], help="Only run these task_ids (default all).")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-output-tokens", type=int, default=12000)
    ap.add_argument("--max-context-tokens", type=int, default=0)
    ap.add_argument("--tokenizer-name-or-path", default="")
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument("--context-token-margin", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy; >0 curbs repetition loops).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    system_prompt = (
        Path(args.system_prompt_file).read_text() if args.system_prompt_file else default_system_prompt()
    )
    tasks = json.loads(Path(args.tasks_file).read_text())
    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [t for t in tasks if t["task_id"] in wanted]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    generation_budgets = {t["task_id"]: args.max_output_tokens for t in tasks}
    if args.max_context_tokens > 0:
        if not args.tokenizer_name_or_path:
            ap.error("--tokenizer-name-or-path is required with --max-context-tokens")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
        chat_template = (
            Path(args.chat_template_file).read_text(encoding="utf-8")
            if args.chat_template_file else None
        )
        prompt_lengths = {}
        for task in tasks:
            rendered_prompt = tokenizer.apply_chat_template(
                task_messages(task, system_prompt),
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_lengths[task["task_id"]] = len(
                tokenizer(rendered_prompt, add_special_tokens=False).input_ids
            )
        for task_id, prompt_tokens in prompt_lengths.items():
            available = args.max_context_tokens - prompt_tokens - args.context_token_margin
            if available < 1:
                raise ValueError(
                    f"{task_id} prompt uses {prompt_tokens} tokens, exceeding the "
                    f"{args.max_context_tokens}-token context window"
                )
            generation_budgets[task_id] = min(args.max_output_tokens, available)
        print(
            f"Prompt tokens: {min(prompt_lengths.values())}-{max(prompt_lengths.values())}; "
            f"generation budgets: {min(generation_budgets.values())}-"
            f"{max(generation_budgets.values())}"
        )

    pred_out = Path(args.pred_out)
    pred_out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and raw_dir is None:
        ap.error("--resume requires --raw-dir")

    results: dict[str, tuple[str, bool]] = {}
    pending_tasks = []
    for task in tasks:
        raw_path = raw_dir / f"{task['task_id']}.txt" if raw_dir else None
        if args.resume and raw_path and raw_path.exists():
            raw = raw_path.read_text(encoding="utf-8")
            if not raw.startswith("[EXC]"):
                prog = extract_program(raw)
                if prog.strip():
                    gold = task["sab"]["gold_program_name"]
                    results[gold] = (prog, True)
                    (pred_out / ("pred_" + gold)).write_text(prog, encoding="utf-8")
                    continue
        pending_tasks.append(task)
    if args.resume:
        print(f"Resume: reused {len(results)} responses; pending {len(pending_tasks)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(one_task, t, args.endpoint, args.model, system_prompt,
                      generation_budgets[t["task_id"]], args.timeout, args.temperature): t["task_id"]
            for t in pending_tasks
        }
        for fut in as_completed(futs):
            task_id = futs.pop(fut)
            gold, prog, ok, raw = fut.result()
            results[gold] = (prog, ok)
            (pred_out / ("pred_" + gold)).write_text(prog, encoding="utf-8")
            if raw_dir:
                (raw_dir / f"{task_id}.txt").write_text(raw, encoding="utf-8")
            tag = "ok" if ok else "FAIL"
            print(f"[{tag}] {task_id} -> pred_{gold} ({len(prog)} chars)")

    # Write ALL tasks in the full tasks file (ERROR placeholder for any not run),
    # so the pred dir + run log line up with the SAB dataset order.
    full = json.loads(Path(args.tasks_file).read_text())
    n_ok = 0
    run_lines = []
    for t in full:
        gold = t["sab"]["gold_program_name"]
        prog, ok = results.get(gold, ("ERROR", False))
        (pred_out / ("pred_" + gold)).write_text(prog, encoding="utf-8")
        n_ok += int(ok)
        run_lines.append(json.dumps({"history": [], "cost": 0.0}))
    if args.run_log:
        Path(args.run_log).write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    print(f"\n生成完成: {n_ok}/{len(tasks)} 有程序 -> {pred_out}"
          + (f"; run log -> {args.run_log}" if args.run_log else ""))


if __name__ == "__main__":
    main()
