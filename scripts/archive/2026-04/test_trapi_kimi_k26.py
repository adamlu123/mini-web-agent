"""Smoke test for the TRAPI Kimi-K2.6_2026-04-20 deployment on gcr/shared."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from miniswewebagent.models.trapi_kimi_model import TrapiKimiModel


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Kimi-K2.6_2026-04-20")
    parser.add_argument("--instance", default="gcr/shared")
    parser.add_argument("--endpoint", default="https://trapi.research.microsoft.com")
    parser.add_argument("--api-version", default="2024-10-21")
    parser.add_argument("--scope", default="api://trapi/.default")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--response-mode", default="none",
                        choices=["none", "json_object", "json_schema"])
    parser.add_argument("--prompt", default="Reply with exactly: pong.")
    args = parser.parse_args()

    model = TrapiKimiModel(
        model_name=args.model,
        trapi_instance=args.instance,
        trapi_endpoint=args.endpoint,
        trapi_api_version=args.api_version,
        trapi_scope=args.scope,
        max_output_tokens=args.max_tokens,
        response_mode=args.response_mode,
        attach_observation_screenshot=False,
    )

    print(f"URL: {model._chat_completions_url()}", file=sys.stderr)

    messages = [
        {"role": "user", "content": args.prompt},
    ]
    payload = model._build_payload(messages)
    print("Request payload:", json.dumps(payload, indent=2), file=sys.stderr)

    import httpx
    token = await model._get_bearer_token()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=model.config.request_timeout_seconds) as client:
        response = await client.post(model._chat_completions_url(),
                                     headers=headers, json=payload)
    print(f"HTTP {response.status_code}", file=sys.stderr)
    try:
        data = response.json()
    except Exception:
        print("Body (non-JSON):", response.text[:2000])
        return 1 if response.status_code >= 400 else 0

    print(json.dumps(data, indent=2))
    if response.status_code >= 400:
        return 1
    choices = data.get("choices") or []
    if not choices:
        print("No choices in response", file=sys.stderr)
        return 1
    content = choices[0].get("message", {}).get("content", "")
    print(f"\n--- content ---\n{content}", file=sys.stderr)
    usage = data.get("usage") or {}
    print(f"--- usage ---\n{json.dumps(usage)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
