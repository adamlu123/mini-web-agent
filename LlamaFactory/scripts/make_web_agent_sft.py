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

A compacted rollout is stored as MULTIPLE sessions: the earlier (pre-compaction)
sessions in `compacted_sessions` and the final session in `messages`. This
converter walks ALL of them in chronological order (compacted first, oldest ->
newest, then the final one), so a rollout of steps 1-10 (compacted) + 11-13
(final) emits the 1-10 session BEFORE the 11-13 session, in order. The previous
version read only `messages` and silently dropped every compacted step (~87% of
all action steps in the N500 set).

Two output shapes via --turn-mode (BOTH are ShareGPT multi-turn):
    multi  (default) -> one full multi-turn sample PER session; train with
                        mask_history=false (loss on EVERY assistant turn).
    single           -> one TRUNCATED multi-turn sample per assistant step t
                        (real human/gpt prefix turns, ending at step t); train
                        with mask_history=true (loss only on the final step t).
                        K steps -> K examples, including the first (cold-start)
                        step of each session. Same total supervised steps as
                        multi, just one step per example.

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


def build_convo(msgs, norm, max_obs):
    """Convert ONE session's message list (system/user/assistant/exit) into a
    clean ShareGPT convo (list of {from: human|gpt, value}). No system field —
    the caller pairs it with the rewritten SYSTEM prompt.

    A `trajectory.json` stores a compacted rollout as MULTIPLE sessions: the
    earlier (pre-compaction) sessions live in `compacted_sessions` and the final
    session in `messages`. Each session is fed here independently so its real
    turns survive (the old converter read only `messages`, silently dropping
    every step that had been compacted away — ~87% of all action steps)."""
    exit_final = ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "exit":
            exit_final = (m.get("extra", {}) or {}).get("final_response", "") or text_of(m.get("content", ""))

    convo = []
    have_final_answer = False
    for m in msgs:
        if not isinstance(m, dict):
            continue
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
    return convo


def iter_sessions(traj: dict):
    """Yield the trajectory's sessions in CHRONOLOGICAL order: the compacted
    (earlier) sessions first, oldest -> newest, then the final live `messages`.
    So a compacted rollout of steps 1-10 (compacted) + 11-13 (final) emits the
    1-10 session BEFORE the 11-13 session — in order, never shuffled."""
    for sess in (traj.get("compacted_sessions") or []):
        if sess:
            yield sess
    msgs = traj.get("messages") or []
    if msgs:
        yield msgs


def convert_traj(traj: dict, normalize: bool, max_obs: int, turn_mode: str):
    """Return a LIST of ShareGPT samples for one trajectory.json.

    turn_mode="multi":  one full multi-turn sample PER session (steps in order);
                        train with mask_history=false (loss on all turns).
    turn_mode="single": one TRUNCATED multi-turn sample per assistant step t
                        (real human/gpt prefix turns ending at step t); train
                        with mask_history=true (loss only on step t). K steps ->
                        K examples, incl. the cold-start first step of each
                        session, using the limited data more finely."""
    ws = (traj.get("environment", {}).get("config", {}) or {}).get("output_dir", "")
    _norm = make_normalizer(ws) if normalize else (lambda s: s)
    # secret redaction ALWAYS runs, independent of path normalization
    norm = lambda s: redact_secrets(_norm(s))

    samples = []
    for sess in iter_sessions(traj):
        convo = build_convo(sess, norm, max_obs)
        if sum(1 for t in convo if t["from"] == "gpt") < 1:
            continue
        if turn_mode == "multi":
            samples.append({"conversations": convo, "system": SYSTEM})
        else:  # single
            # One TRUNCATED multi-turn conversation per assistant step t: keep the
            # real human/gpt role turns of the prefix (NOT flattened into one
            # human turn) so the role structure matches inference, and end at the
            # target step t. Train with mask_history=true so loss falls ONLY on
            # the final gpt turn. convo[0] is always human (build_convo trims
            # leading non-human), so every truncation is a valid human...gpt
            # conversation. Includes the first step of each session (predict the
            # cold-start action from the task / compact summary).
            for i, turn in enumerate(convo):
                if turn["from"] != "gpt":
                    continue
                truncated = [dict(t) for t in convo[: i + 1]]
                samples.append({"conversations": truncated, "system": SYSTEM})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["/data/t-yifeili/sft_data/pae_100"],
                    help="one or more dirs, each containing <task>/trajectory.json "
                         "folders. Multiple dirs are merged in the given order "
                         "(e.g. pae_100 then N500_..._r2_success).")
    ap.add_argument("--out", default="data/web_agent_pae100.json")
    ap.add_argument("--no-normalize-paths", dest="normalize", action="store_false",
                    help="keep raw absolute workspace paths (NOT recommended)")
    ap.add_argument("--max-obs-chars", type=int, default=3000,
                    help="middle-truncate each observation/user turn to this many "
                         "chars (0 = no cap). Keeps long multi-turn convos under "
                         "the trainer cutoff and blunts boilerplate overfitting.")
    ap.add_argument("--turn-mode", choices=["multi", "single"], default="multi",
                    help="multi (default): one multi-turn sample per session, "
                         "sessions emitted in chronological order (compacted "
                         "sessions first, then the final one). single: split "
                         "each session into one single-turn (context -> next "
                         "step) sample per assistant step.")
    args = ap.parse_args()

    # merge multiple source dirs in the given order; sort within each dir for
    # determinism (per-source ordering preserved across sources).
    files = []
    for src in args.src:
        files.extend(sorted(glob.glob(os.path.join(src, "*", "trajectory.json"))))
    data, dropped = [], []
    n_gpt = 0
    n_sessions = 0
    for f in files:
        try:
            traj = json.load(open(f))
        except Exception as e:
            dropped.append((f, f"load error: {e}"))
            continue
        samples = convert_traj(traj, args.normalize, args.max_obs_chars, args.turn_mode)
        if not samples:
            dropped.append((f, "no usable turns"))
            continue
        n_sessions += len(traj.get("compacted_sessions") or []) + 1
        for ex in samples:
            data.append(ex)
            # supervised targets: multi trains every gpt turn (mask_history=false);
            # single trains only the LAST gpt turn per sample (mask_history=true).
            if args.turn_mode == "single":
                n_gpt += 1
            else:
                n_gpt += sum(1 for t in ex["conversations"] if t["from"] == "gpt")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"scanned  {len(files)} trajectory.json ({n_sessions} sessions incl. compacted)")
    print(f"mode     {args.turn_mode}")
    print(f"wrote    {len(data)} samples ({n_gpt} SUPERVISED target steps) -> {args.out}")
    if dropped:
        print(f"dropped  {len(dropped)}:")
        for f, why in dropped:
            print(f"  - {os.path.basename(os.path.dirname(f))}: {why}")


if __name__ == "__main__":
    main()
