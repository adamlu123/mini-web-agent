"""Turn agent trajectories into tokenized SFT examples (assistant-only loss).

Each ``trajectory.json`` produced by the run harness holds a ``messages`` list
(system / user / assistant / exit). Assistant turns carry the exact model target
in ``extra.raw_response`` (the strict JSON action object).

We emit **one example per assistant turn**: the prompt is the conversation history
rendered with the model's chat template (``add_generation_prompt=True``) and the
target is that turn's completion. This is robust to Qwen's template stripping
``<think>`` blocks from non-final assistant turns, since the history always ends on
a user turn and the supervised turn is always rendered as the final one.
"""

import glob
import json
import os
from typing import Dict, List

IGNORE_INDEX = -100


def _norm_content(content) -> str:
    """Flatten a message ``content`` (str or list of {type, text}) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _build_conversation(messages: List[dict]) -> List[Dict[str, str]]:
    """Normalize harness messages into [{role, content}] chat turns.

    Assistant turns use ``extra.raw_response`` (the JSON action) as the target so
    the model learns to emit the same strict-JSON the harness parses. The trailing
    ``exit`` bookkeeping turn is dropped.
    """
    conv: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            raw = (m.get("extra") or {}).get("raw_response")
            text = json.dumps(raw, ensure_ascii=False) if raw else _norm_content(m.get("content"))
            conv.append({"role": "assistant", "content": text})
        elif role in ("system", "user"):
            conv.append({"role": role, "content": _norm_content(m.get("content"))})
        # 'exit' and anything else is ignored
    return conv


def _examples_from_conversation(tokenizer, conv: List[Dict[str, str]], max_len: int):
    """Yield one {input_ids, labels} per assistant turn.

    For assistant turn ``i``: prompt = ``conv[:i]`` with a generation prompt,
    completion = the appended tokens in ``conv[:i+1]``. Sequences longer than
    ``max_len`` are left-truncated so the supervised completion is preserved.
    """
    out = []
    for i, turn in enumerate(conv):
        if turn["role"] != "assistant":
            continue
        prompt_text = tokenizer.apply_chat_template(
            conv[:i], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            conv[: i + 1], tokenize=False, add_generation_prompt=False
        )
        if not full_text.startswith(prompt_text):
            continue  # template not additive for this turn; skip defensively
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        completion_ids = full_ids[len(prompt_ids):]
        if not completion_ids:
            continue
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
        input_ids = list(full_ids)
        # Left-truncate to keep the supervised completion at the tail.
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]
        if not any(l != IGNORE_INDEX for l in labels):
            continue
        out.append({"input_ids": input_ids, "labels": labels})
    return out


def load_examples(data_dir: str, tokenizer, max_len: int = 16384) -> List[Dict[str, list]]:
    """Load every ``*/trajectory.json`` under ``data_dir`` into SFT examples.

    Returns a list of ``{"input_ids": [...], "labels": [...]}``, one per assistant
    turn across all trajectories.
    """
    examples: List[Dict[str, list]] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "trajectory.json"))):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        conv = _build_conversation(data.get("messages", []))
        examples.extend(_examples_from_conversation(tokenizer, conv, max_len))
    return examples


class Collator:
    """Right-pad a batch of {input_ids, labels} to the longest sequence."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, list]]) -> Dict[str, "object"]:
        import torch

        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attn = [], [], []
        for ex in batch:
            n = len(ex["input_ids"])
            pad = max_len - n
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad)
            labels.append(ex["labels"] + [IGNORE_INDEX] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
