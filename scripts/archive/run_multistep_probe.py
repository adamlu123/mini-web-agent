from __future__ import annotations

import json
from pathlib import Path

from miniswewebagent.environments.local_browser import LocalBrowserEnvironment


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "outputs" / "probe-multistep"
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "probe_page.html"
    html_path.write_text(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Local Browser Probe</title>
  </head>
  <body>
    <main>
      <h1 id="title">Counter</h1>
      <p id="value">0</p>
      <button id="increment">Increment</button>
    </main>
    <script>
      const button = document.getElementById('increment');
      const value = document.getElementById('value');
      button.addEventListener('click', () => {
        value.textContent = String(Number(value.textContent) + 1);
        console.log(`counter=${value.textContent}`);
      });
    </script>
  </body>
</html>
        """.strip(),
        encoding="utf-8",
    )

    env = LocalBrowserEnvironment(
        headless=True,
        devtools=False,
        keep_open_on_exit=False,
        prompt_before_close=False,
        slow_mo_ms=0,
        browser_timeout_ms=3000,
        browser_navigation_timeout_ms=3000,
        step_execution_timeout_ms=3000,
        observation_timeout_ms=2000,
        output_dir=output_dir,
    )

    step_records: list[dict[str, object]] = []
    try:
        env.prepare(start_url=html_path.resolve().as_uri(), task="probe", task_id="probe")

        steps = [
            "await page.get_by_role('button', name='Increment').click()",
            "value = await page.locator('#value').text_content()\nassert value == '1', value",
            "await page.get_by_role('button', name='Increment').click()\nvalue = await page.locator('#value').text_content()\nassert value == '2', value",
        ]
        for index, code in enumerate(steps, start=1):
            result = env.execute({"python_code": code})
            step_records.append(
                {
                    "step": index,
                    "code": code,
                    "returncode": result["returncode"],
                    "observation": result["observation"],
                }
            )
    finally:
        env.close()

    (output_dir / "step_log.json").write_text(json.dumps(step_records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()