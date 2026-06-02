#!/usr/bin/env python
"""Generate a small TEXT-ONLY multi-turn sharegpt dataset for smoke-testing the
web-agent SFT pipeline in LlamaFactory.

Each example is a multi-turn conversation:
    system -> human(task+observation) -> gpt(think+action)
           -> human(observation) -> gpt(action) -> ... -> gpt(final answer)

`from: human` = user/observation turns (masked, no loss);
`from: gpt`    = assistant turns (LlamaFactory computes loss on ALL of them by
                 default: mask_history=false, train_on_prompt=false).

Usage:
    python scripts/make_mock_websft.py --n 32 --out data/web_agent_mock.json
"""

import argparse
import json
import random

SYSTEM = (
    "You are a web-browsing agent. You solve the user's task by issuing shell "
    "commands that drive a single live browser tab. Think step by step, then "
    "emit exactly one action per turn as <tool_call>{...}</tool_call>."
)

TASKS = [
    "Find the cheapest direct flight from Seattle to Tokyo next Friday.",
    "Add a 1 kg bag of arabica coffee beans to the cart and go to checkout.",
    "What is the customer rating of the top-listed noise-cancelling headphone?",
    "Book a table for two at an Italian restaurant downtown for 7pm tonight.",
    "Find a vegetarian lasagna recipe with more than 500 reviews.",
    "Locate the return policy window (in days) for electronics on this store.",
]

PAGES = [
    "[page] Search results: 12 items. Top result: 'Acme X100' $129.99, rating 4.3 (812 reviews).",
    "[page] Product detail loaded. Price $129.99. 'Add to cart' button is visible (role=button).",
    "[page] Cart (1 item). Subtotal $129.99. 'Proceed to checkout' link present.",
    "[page] Filters applied. 3 results match. First: 'Nonna's Lasagna' 4.8 stars, 1,203 reviews.",
    "[page] Flight list: AS288 nonstop 10h25m $742; DL167 1 stop $688; NH179 nonstop $815.",
    "[page] Help center article: 'Electronics may be returned within 30 days of delivery.'",
]

ACTIONS = [
    'I will open the top result to inspect details.\n<tool_call>{"name":"click","args":{"role":"link","name":"Acme X100"}}</tool_call>',
    'The add-to-cart button is visible; I will click it.\n<tool_call>{"name":"click","args":{"role":"button","name":"Add to cart"}}</tool_call>',
    'Now proceed to checkout.\n<tool_call>{"name":"click","args":{"role":"link","name":"Proceed to checkout"}}</tool_call>',
    'I will apply the vegetarian filter first.\n<tool_call>{"name":"type","args":{"selector":"#search","text":"vegetarian lasagna"}}</tool_call>',
    'I will sort the flights by price to find the cheapest.\n<tool_call>{"name":"click","args":{"role":"button","name":"Sort by price"}}</tool_call>',
]


def make_example(rng: random.Random) -> dict:
    n_turns = rng.randint(2, 4)  # number of assistant turns
    task = rng.choice(TASKS)
    convo = [{"from": "human", "value": f"Task: {task}\n\n{rng.choice(PAGES)}"}]
    for i in range(n_turns):
        convo.append({"from": "gpt", "value": rng.choice(ACTIONS)})
        if i < n_turns - 1:
            convo.append({"from": "human", "value": rng.choice(PAGES)})
    # final answer turn
    convo.append({"from": "human", "value": rng.choice(PAGES)})
    convo.append(
        {"from": "gpt", "value": f'Based on the page, here is the answer to "{task[:40]}...".\n'
                                 f'<tool_call>{{"name":"submit","args":{{"answer":"<final answer>"}}}}</tool_call>'}
    )
    return {"conversations": convo, "system": SYSTEM}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="data/web_agent_mock.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = [make_example(rng) for _ in range(args.n)]
    with open(args.out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    turns = sum(sum(1 for m in ex["conversations"] if m["from"] == "gpt") for ex in data)
    print(f"wrote {len(data)} convos ({turns} assistant turns total) -> {args.out}")


if __name__ == "__main__":
    main()
