# Claude Opus 5 API client

## Scope

`src/cuawright/models/anthropic_client.py` and `src/cuawright/run/task.py` provide the
first-class Anthropic Messages API transport used by the OSWorld
Actor/subagent runner. Direct Anthropic is selected with
`--model-provider anthropic`; OpenRouter remains available through
`--model-provider openrouter-anthropic`.

The client targets:

```text
claude-opus-5
```

using Anthropic's official Python SDK.

## Design choices

- `AsyncAnthropic` with the official `aiohttp` backend for concurrent agent
  workloads.
- Streaming Messages helper for long, high-effort turns, avoiding idle
  connection failures.
- Adaptive thinking with `display="summarized"`.
- `output_config.effort="max"` for the initial capability-first profile.
- Automatic 5-minute prompt caching through top-level
  `cache_control={"type": "ephemeral"}`.
- Anthropic's beta server-side compaction, using the documented 150K input-token
  trigger and preserving returned `compaction` blocks in conversation history.
- Exact assistant-content round-tripping across tool turns, including thinking
  blocks and encrypted signatures.
- Tool-result-only user turns, with no text appended after `tool_result`.
- Explicit handling for `end_turn`, `tool_use`, `pause_turn`, `max_tokens`,
  `model_context_window_exceeded`, and `refusal`.
- Official SDK typed errors and exponential retries. The client raises the SDK
  errors unchanged so callers can handle authentication, rate limits, and
  service failures specifically.
- Request IDs and usage are exposed on every turn.
- Optional private traces are written mode `0600` and retain full content
  blocks/signatures needed to resume the conversation.
- Every turn has one request owner: `build_create_request` constructs the exact
  payload, the direct-only policy may mutate/prune/preflight that payload, the
  thread traces it, and `create_prepared_turn` sends it without rebuilding it.
- Direct-Anthropic requests permanently replace old model-visible history
  images with the deterministic `[Image Omitted]` marker. Original local trace
  image artifacts are unchanged. Three recent images are always kept, and
  periodic pruning removes four-image batches so prompt-cache prefixes remain
  stable.
- A 24 decimal MB (about 22.9 MiB) serialized-request preflight includes
  messages, nested base64 images, system, tools, thinking, caching, and
  compaction fields. This leaves margin below Anthropic's 32 MB request-body
  limit. When periodic pruning is insufficient, it computes and permanently
  prunes the needed oldest-image prefix in one bounded pass, retaining the
  configured recent minimum, then fails locally if the request is still too
  large. Preflight performs at most three full request serializations.
- Direct desktop images retain up to a 1920-pixel long edge and are compressed
  to at most 1.5 MB. Images whose displayed size differs from the 1920x1080 VM
  desktop include source, displayed, and desktop dimensions plus explicit
  displayed-to-desktop coordinate scale metadata. Original bytes remain in
  trace image artifacts.

## Key handling

The standalone client's default key path is:

```text
~/.claude-key
```

The file must be non-empty and inaccessible to group/other users. The client
rejects permissive modes instead of silently reading them.

The harness launcher defaults to the workspace-root `.a-creds` file for OpenAI
compatibility. For either Claude provider, pass the appropriate key file
explicitly with `--api-key-file`; the launcher does not guess
provider-specific credential locations.

Provider-aware launcher defaults are:

| Provider | Model when omitted | Base URL when omitted |
| --- | --- | --- |
| `openai` | `gpt-5.5` | `https://maif-gateway.azure-api.net/api` |
| `anthropic` | `claude-opus-5` | `https://api.anthropic.com` |
| `openrouter-anthropic` | `anthropic/claude-opus-5` | `https://openrouter.ai/api` |

Direct Anthropic currently accepts only `claude-opus-5` because native cost
accounting is tied to that model's public pricing and fails closed for unknown
prices. OpenRouter obtains exact generation cost independently.

## Smoke and tool-loop test

```bash
.venv/bin/python scripts/python/test_claude_opus5_client.py
```

The test:

1. counts initial input tokens;
2. sends a streaming Opus 5 request with adaptive thinking, high effort, and
   automatic prompt caching;
3. receives a client `tool_use`;
4. executes the local tool;
5. sends a `tool_result` while preserving the complete assistant turn;
6. prints request IDs, stop reasons, thinking summaries, text, tool calls, and
   aggregate usage without printing the key or thinking signatures.

Live validation at `high` effort on 2026-08-23 completed the two-turn tool loop:

```text
37 + 58 -> add_numbers tool -> 95
```

- model: `claude-opus-5`
- stop reasons: `tool_use`, then `end_turn`
- request IDs: present on both streamed turns
- aggregate usage: 482 input, 90 output
- second repeated run: 554 cache-read input tokens
- thinking summary: absent because adaptive thinking skipped reasoning for the
  trivial arithmetic request; this is valid Opus 5 behavior

A later capability-first `max` call could not be validated because the account
credit balance was exhausted before the request reached inference.

To save a resumable private trace:

```bash
.venv/bin/python scripts/python/test_claude_opus5_client.py \
  --private-trace /tmp/claude-opus5-private-trace.json
```

## Production defaults

| Setting | Value |
| --- | --- |
| Model | `claude-opus-5` |
| Effort | `max` |
| Maximum output | 64,000 tokens |
| Thinking | adaptive, summarized display |
| Cache | automatic, ephemeral 5-minute TTL |
| Context management | provider compaction at 150K input tokens |
| Standalone parallel client tools | disabled |
| Harness parallel client tools | enabled and all returned calls processed |
| SDK retries | 4 |
| Timeout | 20 minutes |
| Tool rounds | 64 |
| Direct image history | 3 minimum, prune every 4 images |
| Direct serialized request cap | 24 decimal MB (~22.9 MiB) |
| Direct image preparation | 1920 px long edge, 1.5 MB maximum |

The image-history and payload defaults can be reproduced or overridden with
`--anthropic-min-recent-images`,
`--anthropic-image-prune-interval`, and
`--anthropic-max-request-mb`. These values are captured in each run's
`manifest.json` and affect direct Anthropic only. OpenRouter retains its
existing routing, ZDR, provider-selection, image preparation, and exact
generation-cost behavior.

Anthropic recommends starting Opus 5 at `high` and measuring an effort sweep.
This initial profile deliberately selects `max` for capability-first
experiments. A 64K output ceiling gives thinking and tool arguments sufficient
room; `max_tokens` remains a hard limit across thinking and visible output.

The standalone client defaults `disable_parallel_tool_use=true`. The established
Claude harness intentionally overrides it to `false`, accepts every returned
tool call, and executes the existing multi-call path; prompts tell Claude to
parallelize only independent, non-conflicting work. The GPT-5.5 Responses path
retains its pinned first-tool-only behavior and `parallel_tool_calls=false`.

## Official references

- [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)
- [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [Errors](https://platform.claude.com/docs/en/api/errors)

Before a full scored panel, run a low-width direct-Anthropic protection check
to validate provider behavior and operational limits. The offline unit suite
does not spend API credits and cannot prove live provider availability.
