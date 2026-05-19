"""Web-agent RL: drop-in replacement for the terminal-agent rollout where the
"terminal" drives a Playwright browser instead of a Docker container.

The agent emits the same qwen35-style tool calls as the terminal agent. Each
call is a shell command interpreted by ``WebAgentEnvironment``:

- ``web <subcommand> [args]`` operates the browser via a built-in CLI.
- ``python -c '<code>'`` / ``python <path>`` executes Python with ``page``,
  ``context``, ``browser`` and ``task`` injected.
- Anything else falls back to ``subprocess`` inside the rollout workspace.
"""
