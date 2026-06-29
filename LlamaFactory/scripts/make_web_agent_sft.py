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
from pathlib import Path

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


def workspace_from_traj(traj: dict, traj_path: str | None = None) -> str:
    """Best-effort workspace path across old and new trajectory schemas."""
    candidates = [
        (((traj.get("environment") or {}).get("config") or {}).get("output_dir")),
        ((((traj.get("info") or {}).get("config") or {}).get("environment") or {}).get("output_dir")),
    ]
    agent_output_path = ((((traj.get("info") or {}).get("config") or {}).get("agent") or {}).get("output_path"))
    if agent_output_path:
        candidates.append(str(Path(str(agent_output_path)).parent))
    if traj_path:
        candidates.append(str(Path(traj_path).resolve().parent))
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def assistant_count_in_messages(messages: list[dict]) -> int:
    return sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "assistant")


def is_compact_summary_message(message: dict) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
    if extra.get("interrupt_type") == "HistoryCompactionSummary":
        return True
    return str(text_of(message.get("content", ""))).lstrip().startswith("## Compacted History Summary")


def initial_user_from_steps_md(task_dir: Path) -> str:
    steps_md = task_dir / "debug" / "steps.md"
    if not steps_md.is_file():
        return ""
    text = steps_md.read_text(encoding="utf-8", errors="replace")
    marker = "### Model Input"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = text.find("### Thought", start)
    value = text[start:end if end >= 0 else None].strip()
    return value


def fallback_initial_user(traj: dict, task_dir: Path) -> str:
    value = initial_user_from_steps_md(task_dir)
    if value:
        return value
    task_path = task_dir / "task.json"
    if task_path.is_file():
        try:
            task = json.loads(task_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            task = {}
        parts = []
        if task.get("task"):
            parts.append(f"Task: {task.get('task')}")
        if task.get("task_id"):
            parts.append(f"Task ID: {task.get('task_id')}")
        if task.get("start_url"):
            parts.append(f"Start URL: {task.get('start_url')}")
        if parts:
            return "\n".join(parts)
    for message in traj.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return text_of(message.get("content", ""))
    return "Task continuation."


def original_task_from_initial_user(initial_user: str) -> str:
    for line in str(initial_user or "").splitlines():
        if line.startswith("Task: "):
            return line[len("Task: "):].strip()
    return ""


def parse_raw_model_payload(raw_text: str) -> dict | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    candidates = [text]
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def compact_summaries_from_raw_responses(task_dir: Path, step_numbers: set[int]) -> dict[int, str]:
    raw_path = task_dir / "raw_responses.jsonl"
    if not raw_path.is_file():
        return {}
    summaries: dict[int, str] = {}
    for index, line in enumerate(raw_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if index in step_numbers:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        payload = parse_raw_model_payload(str(row.get("raw_text") or ""))
        if not payload:
            continue
        has_action = bool(str(payload.get("bash_command") or payload.get("python_code") or "").strip())
        if has_action or payload.get("done"):
            continue
        summary = str(payload.get("thought") or payload.get("summary") or payload.get("final_response") or "").strip()
        normalized = " ".join(summary.lower().split())
        if len(normalized) >= 40 and any(
            marker in normalized for marker in ("task goal", "original task", "critical point", "workspace")
        ):
            summaries[index] = summary
    return summaries


def render_compact_summary(*, original_task: str, step_index: int, summary: str) -> str:
    return (
        "## Compacted History Summary\n"
        f"Original task: {original_task}\n"
        f"(context was compacted after step {step_index}; earlier turns have been replaced by the summary below)\n\n"
        f"{summary.strip()}\n\n"
        "## End of Compacted Summary"
    )


def render_observation_from_debug(output: dict) -> str:
    observation = output.get("observation") if isinstance(output.get("observation"), dict) else {}
    success = observation.get("success")
    status = "ok" if success else "error"
    lines = ["Observation:", f"Status: {status}"]
    workspace = observation.get("workspace_dir")
    cwd = observation.get("cwd")
    returncode = output.get("returncode", observation.get("returncode"))
    exception = observation.get("exception") or output.get("exception_info") or ""
    command_output = observation.get("command_output") or output.get("output") or ""
    final_script_path = observation.get("final_script_path") or ""
    if workspace:
        lines.append(f"Workspace: {workspace}")
    if cwd:
        lines.append(f"Working directory: {cwd}")
    if returncode is not None:
        lines.append(f"Return code: {returncode}")
    if exception:
        lines.extend(["Exception:", str(exception)])
    if command_output:
        lines.extend(["Command output:", str(command_output)])
    if final_script_path:
        lines.append(f"final_script.py: {final_script_path}")
    return "\n".join(lines)


def assistant_message_from_debug_step(row: dict) -> dict:
    bash_command = str(row.get("bash_command") or "").strip()
    python_code = str(row.get("python_code") or "").strip()
    command_text = str(row.get("command_text") or bash_command or python_code or "").strip()
    actions = []
    if command_text:
        action = {"command": command_text}
        if bash_command:
            action["bash_command"] = bash_command
        if python_code:
            action["python_code"] = python_code
        actions.append(action)
    return {
        "role": "assistant",
        "content": str(row.get("thought") or ""),
        "extra": {
            "actions": actions,
            "done": bool(row.get("done")),
            "final_response": str(row.get("final_response") or ""),
            "raw_response": row.get("raw_response") or {},
        },
    }


def recover_sessions_from_debug_steps(traj: dict, traj_path: str | None) -> list[list[dict]]:
    if not traj_path or traj.get("compacted_sessions"):
        return []
    task_dir = Path(traj_path).resolve().parent
    steps_dir = task_dir / "debug" / "steps"
    if not steps_dir.is_dir():
        return []
    step_rows: list[tuple[int, dict]] = []
    for step_path in sorted(steps_dir.glob("step_*.json")):
        try:
            row = json.loads(step_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        try:
            step_index = int(row.get("step") or step_path.stem.split("_")[-1])
        except Exception:
            continue
        step_rows.append((step_index, row))
    if not step_rows:
        return []
    messages = traj.get("messages") or []
    current_assistant_count = assistant_count_in_messages(messages)
    first_user_is_summary = any(is_compact_summary_message(message) for message in messages[:3])
    if len(step_rows) <= current_assistant_count and not first_user_is_summary:
        return []

    step_numbers = {step for step, _row in step_rows}
    summaries = compact_summaries_from_raw_responses(task_dir, step_numbers)
    if not summaries and not first_user_is_summary:
        return []

    initial_user = fallback_initial_user(traj, task_dir)
    original_task = original_task_from_initial_user(initial_user)
    groups: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    summary_indices = sorted(summaries)
    summary_cursor = 0
    active_summary_index: int | None = None
    group_start_messages: list[dict] = []

    def start_message_for(summary_index: int | None) -> dict:
        if summary_index is None:
            return {"role": "user", "content": initial_user, "extra": {}}
        return {
            "role": "user",
            "content": render_compact_summary(
                original_task=original_task,
                step_index=summary_index,
                summary=summaries[summary_index],
            ),
            "extra": {"interrupt_type": "HistoryCompactionSummary"},
        }

    group_start_messages.append(start_message_for(None))
    for step_index, row in step_rows:
        while summary_cursor < len(summary_indices) and summary_indices[summary_cursor] < step_index:
            if current:
                groups.append(current)
                current = []
            active_summary_index = summary_indices[summary_cursor]
            group_start_messages.append(start_message_for(active_summary_index))
            summary_cursor += 1
        current.append((step_index, row))
    if current:
        groups.append(current)

    sessions: list[list[dict]] = []
    for index, group in enumerate(groups):
        start_message = group_start_messages[min(index, len(group_start_messages) - 1)]
        session: list[dict] = [start_message]
        for _step_index, row in group:
            session.append(assistant_message_from_debug_step(row))
            for output in row.get("outputs") or []:
                if isinstance(output, dict):
                    session.append({"role": "user", "content": render_observation_from_debug(output), "extra": {}})
        sessions.append(session)
    return sessions


def assistant_value(msg, exit_final, norm):
    """Render ONE assistant turn, or return None to DROP it.

    Cleaning (so the model never trains on malformed targets):
      * placeholder/empty turns (no thought, no command, not a submission) are
        retry/interrupt artifacts -> DROP (otherwise they become empty <think>
        + a spurious mid-trajectory <answer> carrying the final submission).
      * <answer> is emitted ONLY for a genuine terminal turn (done, or the turn
        carries its own final_response). A mid-trajectory no-command turn does
        NOT get exit_final fabricated into a premature <answer>; it is DROPPED.
      * a thought-only turn (no command, no submission) has no actionable target
        -> DROP.
    Dropping these first also removes the consecutive-assistant runs that
    merge_alternation used to fuse into double-<think>/double-<answer> targets."""
    thought = norm(text_of(msg.get("content", ""))).strip()
    extra = msg.get("extra", {}) or {}
    actions = extra.get("actions") or []
    bash_cmds = [norm(a.get("bash_command", "")).strip() for a in actions if a.get("bash_command", "").strip()]
    done = bool(extra.get("done"))
    own_final = norm(extra.get("final_response", "") or "").strip()

    think = f"<think>\n{thought}\n</think>"
    if bash_cmds:
        body = "\n".join(f"<bash>\n{b}\n</bash>" for b in bash_cmds)
        return f"{think}\n{body}"
    # no command: only a GENUINE terminal turn becomes <answer>.
    if done:
        ans = own_final or norm(exit_final or "").strip()
        if not ans:
            return None  # done but empty submission -> nothing to learn
        return f"{think}\n<answer>\n{ans}\n</answer>"
    if own_final:  # not flagged done but carries its own submission -> terminal
        return f"{think}\n<answer>\n{own_final}\n</answer>"
    # no command, not terminal: placeholder / think-only retry artifact -> DROP
    return None


def merge_alternation(convo):
    """Make the convo strictly alternate human/gpt (ShareGPT requirement).
    Consecutive human turns (observations) are concatenated. Consecutive gpt
    turns are a harness artifact (two actions with no observation between); keep
    only the LATEST so we never fuse two <think>/<bash> blocks into one target
    (which is what taught the model to emit multiple <think> per turn)."""
    merged = []
    for turn in convo:
        if merged and merged[-1]["from"] == turn["from"]:
            if turn["from"] == "human":
                merged[-1]["value"] += "\n" + turn["value"]
            else:
                merged[-1] = dict(turn)
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
            if val is None:
                continue  # placeholder / think-only / empty submission -> drop
            convo.append({"from": "gpt", "value": val})
            if "<answer>" in val:
                have_final_answer = True

    # If the rollout never emitted a done turn but the harness recorded a final
    # submission, append it as the closing gpt <answer> turn.
    if not have_final_answer and exit_final and convo and convo[-1]["from"] == "human":
        convo.append({"from": "gpt", "value": f"<think>\nTask complete.\n</think>\n<answer>\n{norm(exit_final).strip()}\n</answer>"})

    # Enforce a SINGLE terminal <answer>: drop any non-final gpt turn that emitted
    # <answer> (a premature/spurious mid-trajectory submission that would teach
    # the model to stop early). Done before merge so it can't strand the answer.
    gpt_positions = [i for i, t in enumerate(convo) if t["from"] == "gpt"]
    if gpt_positions:
        last_gpt = gpt_positions[-1]
        convo = [t for i, t in enumerate(convo)
                 if not (t["from"] == "gpt" and i != last_gpt and "<answer>" in t["value"])]

    convo = merge_alternation(convo)
    # Must start with human and end with gpt; trim offending edges.
    while convo and convo[0]["from"] != "human":
        convo.pop(0)
    while convo and convo[-1]["from"] != "gpt":
        convo.pop()
    return convo


def iter_sessions(traj: dict, traj_path: str | None = None):
    """Yield the trajectory's sessions in CHRONOLOGICAL order: the compacted
    (earlier) sessions first, oldest -> newest, then the final live `messages`.
    So a compacted rollout of steps 1-10 (compacted) + 11-13 (final) emits the
    1-10 session BEFORE the 11-13 session — in order, never shuffled."""
    recovered = recover_sessions_from_debug_steps(traj, traj_path)
    if recovered:
        for sess in recovered:
            if sess:
                yield sess
        return
    for sess in (traj.get("compacted_sessions") or []):
        if sess:
            yield sess
    msgs = traj.get("messages") or []
    if msgs:
        yield msgs


def convert_traj(
    traj: dict,
    normalize: bool,
    max_obs: int,
    turn_mode: str,
    traj_path: str | None = None,
    *,
    return_session_count: bool = False,
):
    """Return a LIST of ShareGPT samples for one trajectory.json.

    turn_mode="multi":  one full multi-turn sample PER session (steps in order);
                        train with mask_history=false (loss on all turns).
    turn_mode="single": one TRUNCATED multi-turn sample per assistant step t
                        (real human/gpt prefix turns ending at step t); train
                        with mask_history=true (loss only on step t). K steps ->
                        K examples, incl. the cold-start first step of each
                        session, using the limited data more finely."""
    ws = workspace_from_traj(traj, traj_path)
    _norm = make_normalizer(ws) if normalize else (lambda s: s)
    # secret redaction ALWAYS runs, independent of path normalization
    norm = lambda s: redact_secrets(_norm(s))

    samples = []
    session_count = 0
    for sess in iter_sessions(traj, traj_path):
        session_count += 1
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
    if return_session_count:
        return samples, session_count
    return samples


def has_self_judge_label(traj_path: str, label: int) -> bool:
    """Return true if any self_reflection judge_result under the task dir has label."""
    task_dir = Path(traj_path).resolve().parent
    for result_path in sorted(task_dir.glob("final_runs/run_*/judge_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if result.get("predicted_label") == label:
            return True
    return False


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
    ap.add_argument("--require-self-judge-label", type=int, choices=[0, 1], default=None,
                    help="only include trajectories whose task dir has at least one "
                         "final_runs/run_*/judge_result.json with this predicted_label")
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
        if args.require_self_judge_label is not None and not has_self_judge_label(f, args.require_self_judge_label):
            dropped.append((f, f"self judge label != {args.require_self_judge_label}"))
            continue
        try:
            traj = json.load(open(f))
        except Exception as e:
            dropped.append((f, f"load error: {e}"))
            continue
        samples, session_count = convert_traj(
            traj,
            args.normalize,
            args.max_obs_chars,
            args.turn_mode,
            f,
            return_session_count=True,
        )
        if not samples:
            dropped.append((f, "no usable turns"))
            continue
        n_sessions += session_count
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
        for f, why in dropped[:50]:
            print(f"  - {os.path.basename(os.path.dirname(f))}: {why}")
        if len(dropped) > 50:
            print(f"  ... {len(dropped) - 50} more")


if __name__ == "__main__":
    main()
