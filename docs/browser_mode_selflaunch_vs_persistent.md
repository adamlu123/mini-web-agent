# 浏览器模式对比：self-launch（自启动）vs persistent-browser（持久浏览器）

mini-web-agent 的 harness 都是同一个交互范式：模型每轮输出一个 JSON（`thought` / `bash_command` / `done` / `final_response`），在本地 workspace 里执行一条 bash 命令（通常是 Playwright heredoc 脚本），再看观测。两种模式的差别只在**浏览器的生命周期归谁管**。

## 1. self-launch 模式（旧，OM2W 4000 采集所用）

代表配置：`src/miniswewebagent/config/best_default_judge_json_agnostic.yaml`
核心工具：`src/miniswewebagent/tools/browser_session.py`（暴露为轨迹无关的顶层模块 `browser_session`）

- **每个 Playwright 步骤都新开一个浏览器 session**。agent 在脚本里 `from browser_session import open_browser_session`，`browser = await open_browser_session(playwright)` 拿到一个新的 CDP 连接，脚本末尾 `await browser.close()` 真正关掉这个 session。
- 后端由 `MWA_BROWSER_BACKEND` 环境变量决定（`browserbase` 云 session / `local` 本地 Chromium / `cdp` 外部端点），agent 代码里完全不出现 provider 细节，保证录出的轨迹是 provider-agnostic 的。
- system prompt 明确写死：**"There is NO persistent browser state"** —— 每次运行必须从 start URL 重新导航、用代码重建全部状态（重新搜索、重新点开 filter、重新选中选项）。
- 因此 agent 的典型写法是把整条操作链压进**一个越写越长的脚本**：探索时不断往同一段代码里追加动作，最终 `final_script.py` 是一个从零可复现整个任务的自包含脚本。
- 产物布局：`final_runs/run_<id>/`（final_script.py、final_script_log.txt、screenshots/），self_reflection 结果写在 `final_runs/run_<id>/judge_result.json`。

**痛点**：状态不可延续。cookie、localStorage、打开的抽屉/下拉框在每步之间全部丢失；每一步都要重放前面所有动作，脚本越来越长、越来越慢，也更容易在重放中途因页面波动失败。

## 2. persistent-browser 模式（新，260718 run_1 所用）

代表配置：`src/miniswewebagent/config/persistent_browser.yaml`
核心工具：`src/miniswewebagent/tools/local_browser_session.py`（persistent-browser 仓库里叫 `webwright.tools.persistent_local_browser`，接口一致）

- **整个 run 只有一个常驻本地 Chromium 子进程**。run 开始时 agent 执行一次
  `create --out .lb_session.json`：以 `--remote-debugging-port=0` + 独立 `--user-data-dir` 拉起一个 detached headless Chromium，把 `{id, pid, connectUrl, userDataDir}` 写进 workspace 的 `.lb_session.json`。
- 之后**每个 Playwright 步骤都是"附着"而不是"启动"**：读 `.lb_session.json`，`playwright.chromium.connect_over_cdp(connectUrl)`，做一个聚焦的小交互，末尾 `await browser.close()` —— 对 CDP 附着的浏览器这只断开 Playwright 连接，Chromium 子进程继续活着。
- 于是页面状态、cookie、localStorage、甚至当前打开的下拉框/弹窗都**跨步骤保留**。探索循环变成"打开页面 → 下一步展开 filter → 下一步勾选 → 下一步截图验证"，每步都是短脚本，不再重放历史。
- `final_script.py` 也附着同一个持久 session（禁止 `chromium.launch(...)`），CLI 还提供 `info`（查 pid 存活）和 `release`（SIGTERM/SIGKILL + 清理 user-data-dir）。
- **完成门槛新增一条**：declare done 之前必须 `release --delete-file --delete-user-data`，且 `.lb_session.json` 已不存在，否则会留僵尸 Chromium。
- 产物布局变化：主轨迹为 `trajectory.json` + `raw_responses.jsonl` + `debug/steps/`；浏览器证据为 `browser-steps.jsonl` + `screenshots/`；反思结果在 `reflection/judge_result.json`（不再是 `final_runs/run_*/judge_result.json`）。

## 3. 对 SFT 数据的影响（对应 persistent WebWright 转换 issue）

两种模式都有 sequential compaction（`agents/default.py` 的 `_compact_history`，`summary_every_n_steps`：旧配置 10 步、persistent 配置 20 步），但数据形态差别很大：

| 维度 | self-launch（旧） | persistent-browser（新） |
|---|---|---|
| 浏览器生命周期 | 每步新开、`browser.close()` 真关 | 整个 run 一个 Chromium，CDP 附着/断开 |
| 状态延续 | 无，每步从头重建 | cookie/页面/控件状态跨步保留 |
| 单步脚本形态 | 长脚本，重放全部历史动作 | 短脚本，只做一个增量交互 |
| 额外义务 | 无 | run 末必须 release，否则完成门槛拒绝 done |
| 会话长度 | 较短 | 明显更长（未截断 median ≈33.9k/42.0k tokens），需 1000 字符 observation 截断才装进 32k |
| compaction 存档 | 只有压缩后的消息 | `compacted_sessions + messages` 保留全部时序窗口，可提取 31 个真实 summary 监督目标 |
| 判据产物 | `final_runs/run_*/judge_result.json` | `reflection/judge_result.json`，且新增 `browser-steps.jsonl` 证据 |
| 转换器 | #26 的旧转换脚本 | `convert_persistent_webagent_to_webwright.py`（新，适配上述布局） |

**一句话总结**：self-launch 把"状态"编码进越来越长的可复现脚本里，轨迹短但每步昂贵且脆；persistent-browser 把状态放进常驻 Chromium，agent 得以做真正的增量探索，轨迹更接近人类操作流，但会话显著变长（32k 下部分行被丢，尤其是 pre-compaction summary 窗口 31 个里只有 1 个装得下），且引入了 session 创建/释放的生命周期纪律。
