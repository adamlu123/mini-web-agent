"""Real one-task rollout: model + Browserbase + osw judge.

End-to-end smoke that exercises every moving part for real (no stubs):

- Loads ``Qwen/Qwen3.5-4B`` with transformers, generates turns directly
  (no vLLM/SkyRL needed) using the qwen35 chat template.
- Drives a live Browserbase Chromium session via ``WebAgentEnvironment``.
- Parses tool calls with the same ``Qwen35XMLParser`` used during training.
- Scores the finished rollout with the real ``OSWJudgeReward`` (o4-mini via
  the phyagi gateway).

Usage::

    source /home/luyadong/sandbox/cred.sh   # BROWSERBASE_*, OPENAI_GATEWAY_*
    python -m echo_rl.web_agent.scripts.run_real_rollout \\
        --task-id smoke-001 \\
        --task "Open https://example.com and report the page title." \\
        --start-url https://example.com \\
        --max-turns 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echo_rl.terminal_agent.parsers import check_format_warnings, get_parser  # noqa: E402
from echo_rl.web_agent.prompts import format_web_task_prompt  # noqa: E402
from echo_rl.web_agent.rewards import build_reward_fn  # noqa: E402
from echo_rl.web_agent.web_environment import (  # noqa: E402
    WebAgentEnvironment,
    WebAgentEnvironmentConfig,
)


def _truncate_obs(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated; original {len(text)} chars]"


class VLLMHttpPolicy:
    """OpenAI-compatible client that hits a remote vLLM server (e.g. Qwen3-VL-4B).

    Same ``generate_turn(messages) -> str`` interface as HFModelPolicy, so
    nothing else in this script changes when you swap policies.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer_path: str | None = None,
        api_key: str = "dummy",
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> None:
        from openai import OpenAI
        from transformers import AutoTokenizer

        # The tokenizer is only used by format_web_task_prompt() to render the
        # system block via apply_chat_template; vLLM itself tokenizes server-side.
        tok_src = tokenizer_path or model
        print(f"[policy] loading tokenizer {tok_src} (for prompt formatting only)...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
        self.model = model
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        print(f"[policy] vLLM endpoint ready: {base_url} model={model}", flush=True)

    def generate_turn(self, messages: list[dict[str, Any]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Parser expects every turn to open with <think>...</think>. Qwen's
        # chat template auto-opens it on the HF path; here we prepend if the
        # model didn't emit it (Qwen3-VL may or may not, depending on version).
        if not text.lstrip().startswith("<think>"):
            text = "<think>\n" + text
        return text.replace("<|im_end|>", "").strip()


class HFModelPolicy:
    """Tiny HF-transformers wrapper that returns one assistant turn at a time."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        print(f"[policy] loading tokenizer {model_path}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        print(f"[policy] loading model {model_path} ({dtype}) on {device}...", flush=True)
        t0 = time.time()
        # cuDNN's fused attention rejects Qwen3.5's MoE head shapes on some
        # driver builds with "No valid execution plans built"; fall back to
        # the math/flash kernels via the SDPA dispatcher.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        print(f"[policy] model ready in {time.time() - t0:.1f}s; im_end_id={self._im_end_id}", flush=True)

    def generate_turn(self, messages: list[dict[str, Any]]) -> str:
        import torch

        # apply_chat_template with add_generation_prompt=True ends the prompt
        # with "<|im_start|>assistant\n<think>" for Qwen3.5; we let the model
        # continue from there and only decode the new tokens.
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._im_end_id,
            )
        new_tokens = output[0, prompt_ids.shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        # The template already opened <think>, so prepend it for parser sanity.
        if not text.lstrip().startswith("<think>"):
            text = "<think>\n" + text
        # Strip trailing <|im_end|> if present so the parser sees clean XML.
        return text.replace("<|im_end|>", "").strip()


async def _async_main(args: argparse.Namespace) -> int:
    if args.endpoint:
        policy = VLLMHttpPolicy(
            base_url=args.endpoint,
            model=args.model,
            tokenizer_path=args.tokenizer or args.model,
            api_key=args.api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        policy = HFModelPolicy(
            model_path=args.model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    parser = get_parser(args.parser)

    # Optional parquet mode: iterate over (task_id, task, start_url) rows and
    # run each one as a fresh rollout. Without --parquet, fall through to the
    # single-task CLI path.
    if args.parquet:
        return await _run_parquet(policy, parser, args)

    return await _run_one_task(
        policy=policy,
        parser=parser,
        task_id=args.task_id,
        task=args.task,
        start_url=args.start_url,
        args=args,
    )


async def _run_one_task(
    *,
    policy,
    parser,
    task_id: str,
    task: str,
    start_url: str,
    args: argparse.Namespace,
) -> int:
    # Build the system + user prompt the same way the dataset does.
    base_messages = format_web_task_prompt(
        task=task,
        start_url=start_url,
        parser_name=args.parser,
        tokenizer=policy.tokenizer,
        add_instruction_prefix=True,
    )
    messages = list(base_messages)

    cfg = WebAgentEnvironmentConfig(
        task_id=task_id,
        task=task,
        start_url=start_url,
        workspace_root=Path(args.workspace_root),
        headless=True,
        use_browserbase=True,
        browserbase_api_key=os.environ.get("BROWSERBASE_API_KEY", ""),
        browserbase_project_id=os.environ.get("BROWSERBASE_PROJECT_ID", ""),
        stub=False,
    )
    env = WebAgentEnvironment(cfg)
    print(f"[env] starting Browserbase session for task {task_id!r}...", flush=True)
    await env.setup()
    print(f"[env] ready; workspace={env.workspace}", flush=True)

    stop_reason = "max_turns"
    final_response = ""
    transcript: list[dict[str, Any]] = []
    transcript_path = Path(args.workspace_root) / f"transcript_{args.task_id}.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush_transcript() -> None:
        # Write + fsync after each new entry so we can `cat` / `jq` the
        # transcript live while the rollout is still running.
        tmp = transcript_path.with_suffix(transcript_path.suffix + ".tmp")
        with tmp.open("w") as fh:
            json.dump(transcript, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(transcript_path)

    try:
        for turn in range(args.max_turns):
            print(f"\n[turn {turn + 1}] generating...", flush=True)
            t0 = time.time()
            response_text = await asyncio.to_thread(policy.generate_turn, messages)
            gen_sec = time.time() - t0
            preview = response_text[:300].replace("\n", " ")
            print(f"[turn {turn + 1}] generated in {gen_sec:.1f}s; preview: {preview!r}", flush=True)

            messages.append({"role": "assistant", "content": response_text})
            transcript.append({"turn": turn + 1, "role": "assistant", "text": response_text, "gen_sec": gen_sec})
            _flush_transcript()

            warnings, violations = check_format_warnings(
                response_text, tags=parser.format_tags, parser_name=args.parser
            )
            parsed = parser.parse_response(response_text)
            if parsed.error and not parsed.is_done:
                obs = (
                    "Format error: " + parsed.error
                    + ("\nWarnings: " + "; ".join(warnings) if warnings else "")
                )
                messages.append({"role": "tool", "content": obs})
                transcript.append({"turn": turn + 1, "role": "tool", "text": obs})
                _flush_transcript()
                continue

            if parsed.is_done and not parsed.commands:
                stop_reason = "done"
                # Look for an explicit "answer:" line in any thought block.
                final_response = parsed.thinking
                print(f"[turn {turn + 1}] agent signalled done.", flush=True)
                break

            cmd_obj = parsed.commands[0]
            if cmd_obj.error or cmd_obj.name != "bash":
                obs = f"Command rejected: {cmd_obj.error or cmd_obj.name}. Only 'bash' is supported."
                messages.append({"role": "tool", "content": obs})
                transcript.append({"turn": turn + 1, "role": "tool", "text": obs})
                _flush_transcript()
                continue

            cmd_str = cmd_obj.arguments.get("command", "").strip()
            timeout = float(cmd_obj.arguments.get("timeout", 30) or 30)
            print(f"[turn {turn + 1}] exec: {cmd_str!r} (timeout={timeout}s)", flush=True)
            result = await env.exec(cmd_str, timeout=timeout)
            tool_text = (result.stdout or "").rstrip()
            if result.stderr:
                tool_text = (tool_text + f"\n[stderr]\n{result.stderr.rstrip()}").strip()
            tool_text = _truncate_obs(tool_text, args.max_obs_chars)
            tool_text += f"\n\n(exit_code={result.return_code})"
            obs_preview = tool_text[:200].replace("\n", " ")
            print(f"[turn {turn + 1}] obs preview: {obs_preview!r}", flush=True)
            messages.append({"role": "tool", "content": tool_text})
            transcript.append({"turn": turn + 1, "role": "tool", "text": tool_text, "exit_code": result.return_code})
            _flush_transcript()
            if parsed.is_done:
                stop_reason = "done"
                break
        else:
            stop_reason = "max_turns"
    finally:
        print(f"\n[env] cleaning up Browserbase session...", flush=True)
        await env.cleanup()

    # Final transcript write (paranoia — should already be on disk).
    _flush_transcript()
    print(f"[transcript] saved {len(transcript)} entries to {transcript_path}", flush=True)

    # Real OSW judge call.
    print(f"\n[judge] scoring trajectory with OSWJudgeReward ({args.judge_model})...", flush=True)
    reward_fn = build_reward_fn(
        {
            "name": "osw_judge",
            "judge_model": args.judge_model,
            "judge_gateway_endpoint": args.judge_endpoint
            or os.environ.get("OPENAI_GATEWAY_ENDPOINT", ""),
            "score_threshold": args.score_threshold,
            "mini_web_agent_root": args.mini_web_agent_root,
        }
    )
    result = await reward_fn.score(
        task=task,
        start_url=start_url,
        actions=env.actions_history(),
        thoughts=[],
        screenshot_paths=env.screenshot_paths(),
        final_response=final_response,
        extra={"task_id": task_id},
    )

    summary = {
        "task_id": task_id,
        "task": task,
        "stop_reason": stop_reason,
        "actions": env.actions_history(),
        "screenshots": env.screenshot_paths(),
        "reward": result.reward,
        "correct": result.correct,
        "judge_error": result.error,
        "judge_metadata": {k: v for k, v in (result.metadata or {}).items() if k not in {"image_judge_record", "response"}},
        "judge_response_preview": (result.metadata or {}).get("response", "")[:600],
    }
    print("\n[result]\n" + json.dumps(summary, indent=2, default=str), flush=True)
    # Persist per-task summary so the parquet loop can aggregate later.
    summary_path = Path(args.workspace_root) / f"summary_{task_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return 0 if not result.error else 2


async def _run_parquet(policy, parser, args: argparse.Namespace) -> int:
    import pandas as pd

    df = pd.read_parquet(args.parquet)
    required = {"task_id", "task", "start_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet {args.parquet} missing required columns: {missing}")
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    print(f"[parquet] running {len(df)} task(s) from {args.parquet}", flush=True)

    aggregate: list[dict[str, Any]] = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        task_id = str(row.task_id)
        task = str(row.task)
        start_url = str(row.start_url)
        print(f"\n========== [{i}/{len(df)}] task_id={task_id} ==========", flush=True)
        try:
            await _run_one_task(
                policy=policy,
                parser=parser,
                task_id=task_id,
                task=task,
                start_url=start_url,
                args=args,
            )
        except Exception as exc:
            print(f"[parquet] task {task_id!r} crashed: {exc!r}", flush=True)
            continue
        # Read back the per-task summary written by _run_one_task.
        summary_path = Path(args.workspace_root) / f"summary_{task_id}.json"
        if summary_path.exists():
            aggregate.append(json.loads(summary_path.read_text()))

    # Aggregate: pass-rate + per-task table.
    total = len(aggregate)
    correct = sum(1 for s in aggregate if s.get("correct"))
    print(f"\n[parquet] {correct}/{total} correct ({(correct/total*100) if total else 0:.1f}%)", flush=True)
    agg_path = Path(args.workspace_root) / "aggregate.json"
    agg_path.write_text(json.dumps({"total": total, "correct": correct, "results": aggregate}, indent=2, default=str))
    print(f"[parquet] aggregate written to {agg_path}", flush=True)
    return 0 if total and correct == total else (0 if total else 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", default="smoke-001")
    p.add_argument("--task", default="Open https://example.com and report the page title.")
    p.add_argument("--start-url", default="https://example.com")
    p.add_argument("--parquet", default="",
                   help="Path to a parquet with columns task_id/task/start_url. When set, "
                        "iterates over all rows instead of using the single --task/--start-url.")
    p.add_argument("--limit", type=int, default=0,
                   help="With --parquet, run only the first N rows. 0 = all.")
    p.add_argument("--parser", default="qwen35", choices=["qwen35", "hermes", "xml"],
                   help="Tool-call format the policy emits. Use 'hermes' for Qwen3-VL (native "
                        "Qwen3 chat-template format).")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B",
                   help="HF model id (HF path) OR vLLM --served-model-name when --endpoint is set.")
    p.add_argument("--endpoint", default="",
                   help="OpenAI-compatible base URL of a remote vLLM server (e.g. http://127.0.0.1:9000/v1). "
                        "When set, the agent talks HTTP and the local HF model is NOT loaded.")
    p.add_argument("--api-key", default="dummy",
                   help="Bearer token for the endpoint; vLLM ignores the value but the client requires non-empty.")
    p.add_argument("--tokenizer", default="",
                   help="HF tokenizer id used only for prompt formatting when --endpoint is set; "
                        "defaults to --model.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-obs-chars", type=int, default=4000)
    p.add_argument("--workspace-root", default="/tmp/web_agent_real_rollout")
    p.add_argument("--judge-model", default="o4-mini")
    p.add_argument(
        "--judge-endpoint", default="",
        help="Defaults to $OPENAI_GATEWAY_ENDPOINT.",
    )
    p.add_argument("--score-threshold", type=int, default=3)
    p.add_argument(
        "--mini-web-agent-root",
        default=os.environ.get("MINI_WEB_AGENT_ROOT", "/home/luyadong/sandbox/mini-web-agent"),
    )
    args = p.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
