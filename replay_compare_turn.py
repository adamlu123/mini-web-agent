"""Replay line i of prompt_messages.jsonl against the policy model and compare
with the corresponding turn in the debug bundle.

Mapping (verified):
  jsonl line -> bundle[example_index], where
    messages == [system] + conversations[:turn_index]  (role user<->human, assistant<->gpt)
    expected output == conversations[turn_index] (a gpt turn)
  Image messages: jsonl uses OpenAI multimodal parts (base64 image_url + text),
  bundle uses an "<image>" placeholder followed by the same text.

Usage:
  python replay_compare_turn.py 0                 # replay line 0, compare input & output
  python replay_compare_turn.py 5 --no-send       # only compare inputs, skip model call
  python replay_compare_turn.py 5 --full          # print full texts without truncation
"""

import argparse
import difflib
import json

PROMPT_JSONL = "/home/yifeili/mini-web-agent/deterministic_replay_m2w_exp_0280/prompt_messages.jsonl"
DEBUG_BUNDLE = "/home/yifeili/mini-web-agent/m2w_exp_0280_debug_bundle/web_agent_debug_m2w_exp_0280.json"

SEP = "=" * 100


def load_line(i):
    with open(PROMPT_JSONL) as f:
        for idx, line in enumerate(f):
            if idx == i:
                return json.loads(line)
    raise IndexError(f"line {i} out of range")


def msg_text(content):
    """Flatten OpenAI message content to comparable text; images -> <image>."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if p.get("type") == "text":
            parts.append(p["text"])
        elif p.get("type") == "image_url":
            parts.append("<image>")
        else:
            parts.append(f"<{p.get('type')}>")
    return "\n".join(parts)


def norm(s):
    """Normalize for comparison: unify <image> placement/whitespace."""
    return s.replace("<image>", "").strip()


def show(label, text, full):
    print(f"----- {label} ({len(text)} chars) -----")
    if full or len(text) <= 3000:
        print(text)
    else:
        print(text[:1500])
        print(f"\n... [{len(text) - 3000} chars omitted, use --full to see all] ...\n")
        print(text[-1500:])
    print()


def diff_report(a, b, name_a, name_b):
    if a == b:
        print(f"  [MATCH] {name_a} == {name_b}")
        return True
    print(f"  [DIFF]  {name_a} != {name_b} (len {len(a)} vs {len(b)})")
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                  fromfile=name_a, tofile=name_b, lineterm="", n=2))
    for line in d[:80]:
        print("    " + line)
    if len(d) > 80:
        print(f"    ... [{len(d) - 80} more diff lines]")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("i", type=int, help="0-based line index into prompt_messages.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="policy")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-send", action="store_true", help="skip the model call, only compare inputs")
    ap.add_argument("--full", action="store_true", help="print full texts without truncation")
    ap.add_argument("--train-render", action="store_true",
                    help="render the prompt exactly as LlamaFactory qwen3_5 training does "
                         "(history assistant turns KEEP their <think> blocks) and call "
                         "/v1/completions instead of /v1/chat/completions. The server chat "
                         "template strips history <think>, which mismatches training; this "
                         "flag reproduces the training-time input. Text-only lines.")
    args = ap.parse_args()

    rec = load_line(args.i)
    messages = rec["messages"]
    ex_i, turn_i = rec["example_index"], rec["turn_index"]

    print(SEP)
    print(f"jsonl line {args.i}: example_index={ex_i} turn_index={turn_i} "
          f"case_type={rec['case_type']} n_messages={len(messages)} "
          f"used_images={len(rec.get('used_images', []))}")
    print(SEP)

    # ---------- full model INPUT text ----------
    print("\n################ MODEL INPUT (messages as sent) ################\n")
    for k, m in enumerate(messages):
        show(f"input msg[{k}] role={m['role']}", msg_text(m["content"]), args.full)

    # ---------- expected input/output from debug bundle ----------
    with open(DEBUG_BUNDLE) as f:
        bundle = json.load(f)
    ex = bundle[ex_i]
    convs = ex["conversations"]

    print("\n################ INPUT COMPARISON vs debug bundle ################\n")
    expected_inputs = [("system", ex["system"])] + [
        ({"human": "user", "gpt": "assistant"}[c["from"]], c["value"]) for c in convs[:turn_i]
    ]
    if len(messages) != len(expected_inputs):
        print(f"  [DIFF] message count: jsonl={len(messages)} bundle={len(expected_inputs)}")
    all_ok = True
    for k, (m, (exp_role, exp_val)) in enumerate(zip(messages, expected_inputs)):
        role_ok = m["role"] == exp_role
        if not role_ok:
            print(f"  msg[{k}]: [DIFF] role jsonl={m['role']} bundle={exp_role}")
        exp_name = "bundle system" if k == 0 else f"bundle conv[{k - 1}]({exp_role})"
        ok = diff_report(norm(msg_text(m["content"])), norm(exp_val),
                         f"jsonl msg[{k}]({m['role']})", exp_name)
        all_ok = all_ok and ok and role_ok
    print(f"\n  => INPUT {'MATCHES' if all_ok else 'DIFFERS FROM'} debug bundle")

    expected_out = convs[turn_i]["value"] if turn_i < len(convs) else None
    print("\n################ EXPECTED OUTPUT (bundle conv[%d], gpt) ################\n" % turn_i)
    if expected_out is None:
        print("  (no gpt turn at this index in bundle)")
    else:
        show("expected output", expected_out, args.full)

    if args.no_send:
        return

    # ---------- call the model ----------
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="dummy")
    if args.train_render:
        if any(not isinstance(m["content"], str) for m in messages):
            raise SystemExit("--train-render only supports text-only lines (this one has images)")
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(f"<|im_start|>system\n{m['content']}<|im_end|>\n")
            elif m["role"] == "user":
                parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>\n<|im_start|>assistant\n")
            else:  # assistant history: keep full <think> block, as in training
                parts.append(f"{m['content']}<|im_end|>\n")
        resp = client.completions.create(
            model=args.model,
            prompt="".join(parts),
            temperature=0,
            max_tokens=args.max_tokens,
            stop=["<|im_end|>"],
        )
        output = resp.choices[0].text
    else:
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0,
            max_tokens=args.max_tokens,
        )
        output = resp.choices[0].message.content

    print("\n################ MODEL OUTPUT ################\n")
    show("model output", output, args.full)

    print("\n################ OUTPUT COMPARISON vs debug bundle ################\n")
    if expected_out is None:
        print("  (nothing to compare)")
    else:
        diff_report(output.strip(), expected_out.strip(), "model output", f"bundle conv[{turn_i}](gpt)")


if __name__ == "__main__":
    main()
