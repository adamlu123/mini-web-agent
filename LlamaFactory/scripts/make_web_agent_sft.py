#!/usr/bin/env python
"""Convert mini-web-agent online rollout trajectories (trajectory.json) into a
multi-turn ShareGPT SFT dataset for LlamaFactory.

Source: a directory of per-task folders, each containing a `trajectory.json`
produced by the online gpt-5.x web agent, e.g.

    sft_data/pae_100/<Task>/trajectory.json

Each trajectory.json is a dict with:
    messages: [ {role, content, extra}, ... ]
        role=system    -> the harness system prompt (DROPPED; we author our own)
        role=user       -> first turn = task + compacted history (string),
                           later turns = observations (list of {type,text} items)
        role=assistant  -> content = the model's "thought";
                           extra.actions[0].bash_command = the shell command;
                           extra.done / extra.final_response on the final turn
        role=exit        -> harness submission (final_response); usually a dup of
                           the last assistant's final_response
    environment.config.output_dir -> the per-task workspace abs path

Target SFT turn formats (ShareGPT, from/value):
    human (task / observation)  ->  kept verbatim (paths normalized)
    gpt   (normal action)       ->  <think>\n{thought}\n</think>\n<bash>\n{cmd}\n</bash>
    gpt   (final / done turn)    ->  <think>\n{thought}\n</think>\n<answer>\n{final}\n</answer>

Anti-overfitting: every task's absolute workspace path
(/home/<user>/sandbox/mini-web-agent/outputs/.../<Task>) is rewritten to a
constant `/workspace`, and the surrounding user-home / repo-root prefixes are
scrubbed, so the small model learns the *structure* of the harness rather than
memorizing one operator's directory layout and the task-name leak. Disable with
--no-normalize-paths.

Usage:
    python scripts/make_web_agent_sft.py \
        --src /data/t-yifeili/sft_data/pae_100 \
        --out data/web_agent_pae100.json
"""

import argparse
import glob
import json
import os
import re

# ---------------------------------------------------------------------------
# Rewritten system prompt: same harness conventions as the online rollouts, but
# the OUTPUT FORMAT section now matches the <think>+<bash>/<answer> SFT target
# instead of the original strict-JSON object. Workspace paths use the same
# `/workspace` placeholder the converter normalizes trajectories to.
# ---------------------------------------------------------------------------
SYSTEM = """You are a benchmark-oriented Web agent operating through a local terminal + workspace harness. Your workspace root is `/workspace`.

Each turn, respond in EXACTLY this format and nothing else:
<think>
your observation, reasoning, and next step
</think>
<bash>
exactly one shell command
</bash>

When (and only when) the task is fully complete and verified, replace the <bash> block with a final answer:
<think>
your final verification reasoning
</think>
<answer>
your final response to the task
</answer>

Global constraints:
- Emit exactly ONE <bash> command per turn (or one <answer> to finish). Never emit raw Python or shell outside the <bash> block. Use heredocs (`python - <<'PY' ... PY`) to run Python inline when needed.
- Reason inside <think>, run one command, then inspect the next Observation before acting again.
- There is NO persistent browser state. Every Playwright run must open a FRESH browser session via the backend-agnostic helper `from browser_session import open_browser_session` (`browser = await open_browser_session(playwright)`), navigate from scratch, and reconstruct state via code. Do NOT hand-write any Browserbase / cloud-provider REST calls or session boilerplate.
- Step screenshots are NOT auto-attached. If you need visual interpretation, invoke the image QA tool yourself: `python -m image_qa --workspace-dir /workspace --image screenshots/foo.png --question "..."`.
- Emit <answer> only AFTER you have executed and verified `final_script.py` and passed self_reflection in a PRIOR turn. Never put a command and a final answer in the same turn.
- Do NOT install packages (pip/apt/etc.). Everything (playwright, httpx, ...) is already installed.

Screenshot rules (HARD):
- Every PNG MUST come directly from `await page.screenshot(path=...)` against a real, unmodified webpage viewport. Images rendered from your own HTML/markdown/summaries, annotated/composited/cropped images, or PIL/matplotlib/HTML-to-image outputs are FAILURES.
- Use a 1280x1800 viewport; never `page.screenshot(full_page=True)`.
- Use stable selectors and current-run evidence. If a control exposes a dedicated filter/sort/style, you must use that control; search terms alone do not satisfy it. Treat numeric/date/quantity/unit constraints as exact. Ground ranking language (best-selling, highest-rated, cheapest, ...) in the site's actual metric.

Workspace artifacts you maintain under `/workspace`:
- `plan.md` — every critical point enumerated as a checklist item.
- `judge_config.json` — the four self_reflection prompts (image_judge_system/user, final_verdict_system/user) authored ONCE and reused.
- `final_script.py` — the deterministic Playwright script; prefer incremental edits once it exists.
- `final_runs/run_<id>/` — each successful execution writes `final_script_log.txt`, screenshots, and (after judging) `judge_result.json`. State the final datum on a `Final Response:` line in `final_script_log.txt`.

Completion gate — finish with <answer> ONLY when all hold:
1. `plan.md` exists with the full critical-point checklist.
2. `judge_config.json` exists with all four prompts populated.
3. `final_script.py` ran from scratch into a fresh `final_runs/run_<id>/`, producing the log and all critical-point screenshots.
4. `python -m self_reflection --config /workspace/judge_config.json --workspace-dir /workspace --output final_runs/run_<id>/judge_result.json` exited 0 with `"predicted_label": 1`.
5. You inspected the run folder (`ls -R final_runs/run_<id>`, the screenshots, and `final_script_log.txt`) and confirmed the artifacts.
If self_reflection fails, diagnose the specific issue, fix `final_script.py`, re-run it in a new `final_runs/run_<id+1>/`, and re-judge — do not edit `judge_config.json` unless a prompt is objectively wrong."""


def make_normalizer(ws: str):
    """Return a fn that scrubs the per-task workspace path + operator home/repo
    prefixes from any string, so trajectories don't leak one machine's layout
    (and the task name embedded in the path)."""
    repls = []
    if ws:
        repls.append((ws, "/workspace"))
        idx = ws.find("mini-web-agent")
        if idx >= 0:
            repls.append((ws[: idx + len("mini-web-agent")], "/opt/mini-web-agent"))
        m = re.match(r"/home/([^/]+)", ws)
        if m:
            user = m.group(1)
            repls.append((f"/home/{user}", "/home/agent"))
            # bare username tokens leak via `ls -la` owner columns, log paths,
            # etc.; scrub them too (applied after the longer path repls below).
            repls.append((user, "user"))
    # longest source first so the most specific path wins
    repls.sort(key=lambda kv: len(kv[0]), reverse=True)

    def norm(s):
        if not isinstance(s, str):
            return s
        for src, dst in repls:
            s = s.replace(src, dst)
        return s

    return norm


# API keys / tokens leaked into observations (e.g. an `env` or `cat cred.sh`
# dump). Redact them so they never reach the SFT dataset. NOTE: the secrets are
# still present in the raw trajectory.json sources — rotate those keys.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),          # OpenAI / Anthropic / OpenRouter (sk-, sk-ant-, sk-proj-, sk-or-)
    re.compile(r"bb_(?:live|test)_[A-Za-z0-9_\-]{10,}"),  # Browserbase
    re.compile(r"AKIA[0-9A-Z]{16}"),                # AWS access key id
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),      # GitHub tokens
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),    # Slack tokens
    re.compile(r"hf_[A-Za-z0-9]{20,}"),             # HuggingFace tokens
]


def redact_secrets(s: str) -> str:
    if not isinstance(s, str):
        return s
    for pat in SECRET_PATTERNS:
        s = pat.sub("<REDACTED_SECRET>", s)
    return s


def cap_obs(s: str, max_chars: int) -> str:
    """Middle-truncate a long observation, keeping the head (Observation header +
    command + start of output) and the tail (final lines / Final Response)."""
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head = int(max_chars * 0.6)
    tail = max_chars - head
    omitted = len(s) - max_chars
    return f"{s[:head]}\n... [{omitted} chars truncated] ...\n{s[-tail:]}"


def text_of(content) -> str:
    """Flatten a message content (string OR list of {type,text/...} items)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for it in content:
            if isinstance(it, dict):
                if "text" in it:
                    parts.append(it["text"])
                elif str(it.get("type", "")).endswith("image"):
                    parts.append("[image omitted]")
            elif isinstance(it, str):
                parts.append(it)
        return "\n".join(parts)
    return str(content)


def assistant_value(msg, exit_final, norm) -> str:
    thought = norm(text_of(msg.get("content", ""))).strip()
    extra = msg.get("extra", {}) or {}
    actions = extra.get("actions") or []
    bash_cmds = [norm(a.get("bash_command", "")).strip() for a in actions if a.get("bash_command", "").strip()]
    think = f"<think>\n{thought}\n</think>"
    done = bool(extra.get("done"))
    if done or not bash_cmds:
        final = norm(extra.get("final_response", "") or exit_final or "").strip()
        return f"{think}\n<answer>\n{final}\n</answer>"
    body = "\n".join(f"<bash>\n{b}\n</bash>" for b in bash_cmds)
    return f"{think}\n{body}"


def merge_alternation(convo):
    """Collapse consecutive same-role turns so the convo strictly alternates
    human/gpt (LlamaFactory's ShareGPT converter requires it)."""
    merged = []
    for turn in convo:
        if merged and merged[-1]["from"] == turn["from"]:
            merged[-1]["value"] += "\n" + turn["value"]
        else:
            merged.append(dict(turn))
    return merged


def convert(traj: dict, normalize: bool, max_obs: int):
    msgs = traj.get("messages", [])
    if not msgs:
        return None
    ws = (traj.get("environment", {}).get("config", {}) or {}).get("output_dir", "")
    _norm = make_normalizer(ws) if normalize else (lambda s: s)
    # secret redaction ALWAYS runs, independent of path normalization
    norm = lambda s: redact_secrets(_norm(s))

    exit_final = ""
    for m in msgs:
        if m.get("role") == "exit":
            exit_final = (m.get("extra", {}) or {}).get("final_response", "") or text_of(m.get("content", ""))

    convo = []
    have_final_answer = False
    for m in msgs:
        role = m.get("role")
        if role in ("system", "exit"):
            continue
        if role == "user":
            val = cap_obs(norm(text_of(m.get("content", ""))).strip(), max_obs)
            if val:
                convo.append({"from": "human", "value": val})
        elif role == "assistant":
            val = assistant_value(m, exit_final, norm)
            convo.append({"from": "gpt", "value": val})
            if "<answer>" in val:
                have_final_answer = True

    # If the rollout never emitted a done turn but the harness recorded a final
    # submission, append it as the closing gpt <answer> turn.
    if not have_final_answer and exit_final and convo and convo[-1]["from"] == "human":
        convo.append({"from": "gpt", "value": f"<think>\nTask complete.\n</think>\n<answer>\n{norm(exit_final).strip()}\n</answer>"})

    convo = merge_alternation(convo)
    # Must start with human and end with gpt; trim offending edges.
    while convo and convo[0]["from"] != "human":
        convo.pop(0)
    while convo and convo[-1]["from"] != "gpt":
        convo.pop()
    if len([t for t in convo if t["from"] == "gpt"]) < 1 or not convo:
        return None
    return {"conversations": convo, "system": SYSTEM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/t-yifeili/sft_data/pae_100",
                    help="dir containing <task>/trajectory.json folders")
    ap.add_argument("--out", default="data/web_agent_pae100.json")
    ap.add_argument("--no-normalize-paths", dest="normalize", action="store_false",
                    help="keep raw absolute workspace paths (NOT recommended)")
    ap.add_argument("--max-obs-chars", type=int, default=3000,
                    help="middle-truncate each observation/user turn to this many "
                         "chars (0 = no cap). Keeps long multi-turn convos under "
                         "the trainer cutoff and blunts boilerplate overfitting.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*", "trajectory.json")))
    data, dropped = [], []
    n_gpt = 0
    for f in files:
        try:
            traj = json.load(open(f))
        except Exception as e:
            dropped.append((f, f"load error: {e}"))
            continue
        ex = convert(traj, args.normalize, args.max_obs_chars)
        if ex is None:
            dropped.append((f, "no usable turns"))
            continue
        data.append(ex)
        n_gpt += sum(1 for t in ex["conversations"] if t["from"] == "gpt")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"scanned {len(files)} trajectory.json")
    print(f"wrote   {len(data)} convos ({n_gpt} assistant turns) -> {args.out}")
    if dropped:
        print(f"dropped {len(dropped)}:")
        for f, why in dropped:
            print(f"  - {os.path.basename(os.path.dirname(f))}: {why}")


if __name__ == "__main__":
    main()
