#!/usr/bin/env python3
"""Archived utility for replaying web-agent SFT data one assistant turn at a time.

For every GPT turn in each ShareGPT example, this script sends the exact gold
prefix before that turn to an OpenAI-compatible chat-completions endpoint and
compares deterministic generation with the gold target. Image examples are sent
as data-url image parts by resolving JSON image paths under --media-dir.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE = "<image>"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = norm(pred).split()
    gold_tokens = norm(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def tag(text: str, name: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def done_value(text: str) -> str:
    match = re.search(r"<done>\s*(true|false)\s*</done>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).lower() if match else ""


def fields(text: str) -> dict[str, str]:
    return {
        "think": tag(text, "think"),
        "bash": tag(text, "bash"),
        "done": done_value(text),
        "final_response": tag(text, "final_response"),
        "answer": tag(text, "answer"),
    }


def field_exact(pred: str, gold: str, name: str) -> bool:
    return norm(fields(pred).get(name, "")) == norm(fields(gold).get(name, ""))


def classify(example: dict[str, Any], turn_index: int) -> str:
    aux = str(example.get("aux_type") or "unknown")
    if aux != "trajectory_session":
        return aux

    conv = example.get("conversations") or []
    if example.get("has_summary_target") and turn_index == len(conv) - 1:
        return "compact_summary"

    first_gpt = next((i for i, t in enumerate(conv) if isinstance(t, dict) and t.get("from") == "gpt"), None)
    if first_gpt == turn_index and int(example.get("window_start_call") or 1) > 1:
        return "compact_to_main_agent"

    return "main_agent_next_turn"


def resolve_media(raw: str, media_dir: Path | None) -> Path | None:
    raw_path = str(raw or "")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    if media_dir is not None:
        candidates.append(media_dir.expanduser() / raw_path)
    candidates.append(path)

    seen: set[str] = set()
    for candidate in candidates:
        key = os.fspath(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def image_part(path: Path) -> dict[str, Any]:
    mime, _ = mimetypes.guess_type(os.fspath(path))
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def content_with_images(
    text: str,
    *,
    raw_images: list[str],
    cursor: int,
    media_dir: Path | None,
) -> tuple[str | list[dict[str, Any]], int, list[str], list[str]]:
    if IMAGE not in text:
        return text, cursor, [], []

    parts: list[dict[str, Any]] = []
    used: list[str] = []
    missing: list[str] = []
    chunks = text.split(IMAGE)
    for i, chunk in enumerate(chunks):
        if chunk:
            parts.append({"type": "text", "text": chunk})
        if i == len(chunks) - 1:
            continue
        if cursor >= len(raw_images):
            missing.append(f"missing image for placeholder {i + 1}")
            continue
        raw = raw_images[cursor]
        cursor += 1
        resolved = resolve_media(raw, media_dir)
        if resolved is None:
            missing.append(raw)
            continue
        used.append(os.fspath(resolved))
        parts.append(image_part(resolved))
    return parts, cursor, used, missing


def prefix_messages(
    example: dict[str, Any],
    *,
    target_turn_index: int,
    media_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    messages: list[dict[str, Any]] = []
    system = str(example.get("system") or "")
    if system:
        messages.append({"role": "system", "content": system})

    raw_images = [str(p) for p in (example.get("images") or [])]
    cursor = 0
    used_images: list[str] = []
    missing_images: list[str] = []

    for turn in (example.get("conversations") or [])[:target_turn_index]:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") == "human":
            content, cursor, used, missing = content_with_images(
                str(turn.get("value") or ""),
                raw_images=raw_images,
                cursor=cursor,
                media_dir=media_dir,
            )
            messages.append({"role": "user", "content": content})
            used_images.extend(used)
            missing_images.extend(missing)
        elif turn.get("from") == "gpt":
            messages.append({"role": "assistant", "content": str(turn.get("value") or "")})
    return messages, used_images, missing_images


def build_cases(data: list[dict[str, Any]], media_dir: Path | None, allowed: set[str] | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for ex_i, example in enumerate(data):
        assistant_i = 0
        for turn_i, turn in enumerate(example.get("conversations") or []):
            if not isinstance(turn, dict) or turn.get("from") != "gpt":
                continue
            case_type = classify(example, turn_i)
            if allowed is not None and case_type not in allowed:
                assistant_i += 1
                continue
            messages, used, missing = prefix_messages(example, target_turn_index=turn_i, media_dir=media_dir)
            cases.append(
                {
                    "example_index": ex_i,
                    "turn_index": turn_i,
                    "assistant_turn_index": assistant_i,
                    "case_type": case_type,
                    "aux_type": example.get("aux_type"),
                    "source": example.get("source"),
                    "window_start_call": example.get("window_start_call"),
                    "window_end_call": example.get("window_end_call"),
                    "summary_raw_index": example.get("summary_raw_index"),
                    "image_index": example.get("image_index"),
                    "predicted_label": example.get("predicted_label"),
                    "messages": messages,
                    "used_images": used,
                    "missing_images": missing,
                    "gold": str(turn.get("value") or ""),
                }
            )
            assistant_i += 1
    return cases


def post_chat(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body[:4000]}") from exc


def response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content") or ""
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return str(content)


def make_record(case: dict[str, Any], pred: str, payload: dict[str, Any]) -> dict[str, Any]:
    gold = case["gold"]
    record = {k: case.get(k) for k in [
        "example_index",
        "turn_index",
        "assistant_turn_index",
        "case_type",
        "aux_type",
        "source",
        "window_start_call",
        "window_end_call",
        "summary_raw_index",
        "image_index",
        "predicted_label",
    ]}
    record.update(
        {
            "used_images": case["used_images"],
            "missing_images": case["missing_images"],
            "n_messages": len(case["messages"]),
            "gold": gold,
            "pred": pred,
            "raw_exact": pred.strip() == gold.strip(),
            "norm_exact": norm(pred) == norm(gold),
            "token_f1": token_f1(pred, gold),
            "think_exact": field_exact(pred, gold, "think"),
            "bash_exact": field_exact(pred, gold, "bash"),
            "done_exact": field_exact(pred, gold, "done"),
            "final_response_exact": field_exact(pred, gold, "final_response"),
            "answer_exact": field_exact(pred, gold, "answer"),
            "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
        }
    )
    return record


def summarize(records: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["case_type"])].append(record)

    metrics = [
        "raw_exact",
        "norm_exact",
        "token_f1",
        "think_exact",
        "bash_exact",
        "done_exact",
        "final_response_exact",
        "answer_exact",
    ]
    by_case: dict[str, Any] = {}
    for case_type, group in sorted(groups.items()):
        by_case[case_type] = {"count": len(group)}
        for metric in metrics:
            by_case[case_type][metric] = sum(float(r[metric]) for r in group) / len(group)

    return {
        "n_scored": len(records),
        "n_skipped": len(skipped),
        "skipped_by_reason": dict(Counter(str(s.get("reason") or "unknown") for s in skipped)),
        "by_case_type": by_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_COMPATIBLE_API_KEY", "dummy"))
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--case-type", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--save-prompts", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--data must point to a JSON list")

    allowed = set(args.case_type) if args.case_type else None
    cases = build_cases(data, args.media_dir, allowed)
    if args.start_index:
        cases = cases[args.start_index:]
    if args.max_cases:
        cases = cases[: args.max_cases]

    print(f"[replay] examples={len(data)} cases={len(cases)} endpoint={args.endpoint} model={args.model}", flush=True)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    prompt_lines: list[str] = []

    for i, case in enumerate(cases, start=1):
        label = f"[replay] {i}/{len(cases)} case={case['case_type']} ex={case['example_index']} turn={case['turn_index']}"
        if case["missing_images"]:
            print(f"{label} skipped missing_images={case['missing_images'][:3]}", flush=True)
            skipped.append({**{k: case.get(k) for k in ("example_index", "turn_index", "case_type")}, "reason": "missing_images", "missing_images": case["missing_images"]})
            continue

        if args.save_prompts:
            prompt_lines.append(json.dumps({k: case[k] for k in ("example_index", "turn_index", "case_type", "used_images", "messages")}, ensure_ascii=False) + "\n")

        payload = post_chat(
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            messages=case["messages"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            timeout=args.timeout,
        )
        pred = response_text(payload)
        record = make_record(case, pred, payload)
        records.append(record)
        print(f"{label} norm_exact={record['norm_exact']} token_f1={record['token_f1']:.3f}", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    (out_dir / "per_turn_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out_dir / "skipped.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in skipped), encoding="utf-8")
    if args.save_prompts:
        (out_dir / "prompt_messages.jsonl").write_text("".join(prompt_lines), encoding="utf-8")

    summary = summarize(records, skipped)
    summary.update(
        {
            "data": args.data,
            "endpoint": args.endpoint,
            "model": args.model,
            "media_dir": os.fspath(args.media_dir) if args.media_dir else None,
            "case_type_filter": sorted(allowed) if allowed else None,
            "start_index": args.start_index,
            "max_cases": args.max_cases,
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
