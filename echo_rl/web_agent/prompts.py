from typing import Any

WEB_AGENT_INSTRUCTION_PREFIX = """You are a web-browsing agent. You solve the user's task by issuing shell commands inside a sandbox that exposes a `web` CLI bound to a single live Playwright browser tab. State persists across commands within one rollout.

Start each response with a <think>...</think> section where you analyze the current browser state from the latest observation and describe your next step. Then emit exactly one bash tool call that runs ONE shell command. When the task is finished, emit a done tool call instead.

Required tools:
- "bash": Execute ONE shell command in the sandbox and return the terminal output (which for `web` commands includes the page URL, title, and an ARIA snapshot of the new page state). Example: <tool_call>\n<function=bash>\n<parameter=command>\nweb goto https://example.com
</parameter>\n<parameter=timeout>\n30
</parameter>\n</function>\n</tool_call>

Optional tools:
- "done": Call exactly once when you have verified the task is complete. Example: <tool_call>\n<function=done>\n</function>\n</tool_call>

The `web` CLI supports these subcommands (one per shell command, no chaining):
- `web goto <url>`                          navigate the tab to <url>
- `web url`                                 print the current URL
- `web title`                               print the page title
- `web snapshot [max_chars]`                print the ARIA accessibility tree (default 6000 chars)
- `web text [max_chars]`                    print visible body text (default 4000 chars)
- `web click <selector>`                    click a Playwright selector (CSS, text=..., role=..., etc.)
- `web fill <selector> <value...>`          fill an input with the given value
- `web press <key>`                         press a keyboard key (e.g. Enter, ArrowDown)
- `web wait <selector> [timeout_ms]`        wait until a selector is visible
- `web back` / `web forward` / `web reload` browser history navigation
- `web screenshot [name]`                   capture a screenshot (auto-named per turn if omitted)
- `web eval <js_expression>`                evaluate a JavaScript expression and print the result
- `web py <python_code>`                    run a Python snippet that has `page`, `context`, `browser`, `task` injected

You can also run plain shell utilities (e.g. `cat`, `ls`, `grep`, `python -c '...'`). Each command runs in a workspace directory at /tmp/web_agent_workspace.

IMPORTANT:
- Only use <think> and <tool_call> tags as described above.
- Exactly one <think>...</think> block per response. Failure to follow the response format will lead to parsing errors.
- Each command runs against the SAME browser tab. Do not relaunch the browser.
- Do not include extra whitespace before or after the command unless intentional.
- Call `done` only after you have verified that the task is fully solved by inspecting the page state."""


def get_instance_template() -> str:
    return (
        "Task: {task}\n\n"
        "Starting URL: {start_url}\n\n"
        "Begin by inspecting the page with `web snapshot`. Then plan and execute the steps. "
        "Call `done` only after the page state confirms the task is complete."
    )


def format_web_task_prompt(
    task: str,
    start_url: str,
    parser_name: str,
    tokenizer: Any,
    add_instruction_prefix: bool = True,
) -> list[dict]:
    if parser_name not in ("qwen35", "hermes"):
        raise ValueError(f"web_agent only supports qwen35 and hermes parsers, got {parser_name!r}")
    from echo_rl.terminal_agent.tools import get_augmented_system_content

    user_content = get_instance_template().format(task=task, start_url=start_url)
    if add_instruction_prefix:
        user_content = WEB_AGENT_INSTRUCTION_PREFIX + "\n\n" + user_content
    return [
        {"role": "system", "content": get_augmented_system_content(tokenizer, parser_name)},
        {"role": "user", "content": user_content},
    ]
