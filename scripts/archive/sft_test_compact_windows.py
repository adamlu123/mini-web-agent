#!/usr/bin/env python3
"""Archived teacher-forced compact-window test for web-agent SFT checkpoints.

For each compact-window ShareGPT sample, condition on the gold prefix ending at
the final summary prompt and greedily generate only the final assistant turn.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _extract_summary(text: str) -> str:
    raw = str(text or "")
    think = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    if think:
        return think[-1].strip()
    if "</think>" in raw.lower():
        return re.split(r"</think>", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return raw.strip()


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ckpt = os.environ["CKPT"]
    data_path = os.environ["DATA"]
    out_dir = Path(os.environ["OUT"])
    max_prompt = int(os.environ.get("MAX_PROMPT_TOKENS", "40000"))
    max_new = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
    gpu_util = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
    tp = int(os.environ.get("TP", "1"))

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))

    samples = []
    skipped_long = 0
    for index, example in enumerate(data):
        conversations = example["conversations"]
        if not conversations or conversations[-1].get("from") != "gpt":
            continue
        system = example.get("system", "")
        messages = [{"role": "system", "content": system}] if system else []
        for turn in conversations[:-1]:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_prompt > max_prompt:
            skipped_long += 1
            continue
        prompt_has_open_think = prompt.rstrip().endswith("<think>")
        samples.append((index, prompt, conversations[-1]["value"], n_prompt, prompt_has_open_think, example))

    print(f"[compact-test] {len(data)} examples -> {len(samples)} scored ({skipped_long} skipped)", flush=True)
    llm = LLM(
        model=ckpt,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_prompt + max_new,
        gpu_memory_utilization=gpu_util,
        tensor_parallel_size=tp,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    outputs = llm.generate([sample[1] for sample in samples], sampling)

    records = []
    exact = 0
    tag_ok = 0
    f1s = []
    for (index, prompt, gold, n_prompt, prompt_has_open_think, example), output in zip(samples, outputs):
        pred = output.outputs[0].text
        pred_summary = _extract_summary(pred)
        gold_summary = _extract_summary(gold)
        f1 = _token_f1(pred_summary, gold_summary)
        f1s.append(f1)
        is_exact = _normalize(pred_summary) == _normalize(gold_summary)
        has_open_think = "<think>" in pred or prompt_has_open_think
        has_tags = has_open_think and all(
            tag in pred for tag in ["</think>", "<bash>", "</bash>", "<done>false</done>", "<final_response>"]
        )
        exact += int(is_exact)
        tag_ok += int(has_tags)
        records.append(
            {
                "example": index,
                "window_start_call": example.get("window_start_call"),
                "window_end_call": example.get("window_end_call"),
                "prompt_tokens": n_prompt,
                "tag_ok": has_tags,
                "summary_exact": is_exact,
                "summary_token_f1": f1,
                "gold": gold,
                "pred": pred,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_example_predictions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "ckpt": ckpt,
        "data": data_path,
        "n_examples": len(data),
        "n_scored": len(samples),
        "n_skipped_long": skipped_long,
        "tag_ok_rate": f"{100 * tag_ok / len(samples):.1f}%" if samples else "n/a",
        "summary_exact_rate": f"{100 * exact / len(samples):.1f}%" if samples else "n/a",
        "avg_summary_token_f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
