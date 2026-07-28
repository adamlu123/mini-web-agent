# web_agent_seq_om2w4000_run4 轨迹数据说明

> run23 之后的最后一个增量批：run4 重跑 run3 中被预算 429 打断的 218 个任务，只取 trajectory_session。
> 生成日期：2026-07-13，转换脚本与参数和 run1/run23 完全一致（`make_web_agent_sequential_compact_tools_sft.py` 默认参数，prompt-mode `sft_state_debug`，label=1 过滤）。格式细节见 `docs/web_agent_seq_om2w4000_run1_dataset.md`，run23 见 `docs/web_agent_seq_om2w4000_run23_dataset.md`。
> **至此 om2w_4000（3149 任务）采集收官**：run1–4 累计 Submitted 2313 条。

## 1. 数据集与路径

| 数据集 | 条数 | 本机路径 | 集群路径（PVC） |
|---|---|---|---|
| run4 纯轨迹 (for OPSD) | 504 | `LlamaFactory/data/web_agent_seq_om2w4000_run4_traj_only.json`（43M，无图片） | `/mnt/pvc/t-yifeili/data/web_agent_seq_om2w4000_run4_traj_only_portable_bundle` |
| **合并版 (for SFT，训练推荐)** | **11403** | `LlamaFactory/data/web_agent_seq_om2w4000_run1r0_plus_run234traj_portable_bundle/`（5.9G） | `/mnt/pvc/t-yifeili/data/web_agent_seq_om2w4000_run1r0_plus_run234traj_portable_bundle` |

- **合并版** = run1 with_reflect0 全量 8774 + run23 纯轨迹 2125 + run4 纯轨迹 504。图片 9825 张 / 引用 11093 处 / missing 0（图片全部来自 run1 部分，run23/run4 轨迹无图）。注册名 `web_agent_seq_om2w4000_run1r0_plus_run234traj_portable`（bundle 内自带 dataset_info.json），训练 yaml 的 `dataset_dir`/`media_dir` 都指 bundle 根。**取代 run23 文档里的 run1r0_plus_run23traj 合并版**。
- 集群版由 PVC 端 job 就地合并生成（json 拼接 + 图片硬链），与本机版内容一致。

## 2. 来源与过滤

- 采集：run4（2026-07-13 下午，新 key bd3abb69 前缀）重跑 run3 的 218 条 RateLimitError 任务，100 并发。结果 215 出 result.json = **127 Submitted / 88 step 耗尽 / 0 限流**；另 3 条卡死无果（judge 全 failure，直接舍弃）。
- 任务筛选：取有 result.json 的 215 个，与 run1/run23 已用 trajectory 任务（2166 个）比对**零重叠**（本批任务在 run23 转换时就是按 ratelimit 整批排除的）。
- label=1 过滤后 **127 个任务**进入数据集（正好 = Submitted 数），切成 504 条窗口样本（377 带 summary 目标 / 127 不带）。
- 已校验：window-0（127 条）全部以 `Task:` 开头，后续窗口（377 条）全部以 `## Compacted History Summary` 开头。

## 3. 统计（Qwen3.5-9B tokenizer，口径同 run1 文档 §4；504 条 trajectory_session）

整条轨迹（127 条）：实际步数 avg 35 / p90 70 / max 96；训练 gpt 轮 avg 32 / p90 62 / max 86。
（比 run1/run23 更长——本批全是 run3 里跑得慢、被 429 打断的偏难任务。）

每条 training example（**output 为 example 内全部 gpt 轮加总**，平均 8.2 轮，非单轮）：

| 指标 | avg | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| example 总 tokens（全轮加总） | 22,553 | 21,474 | 28,869 | 36,071 | 57,049 | 65,362 |
| example output tokens（gpt 轮加总） | 6,497 | 6,791 | 8,957 | 10,688 | 13,763 | 20,619 |

每条 example 的 gpt 轮数 avg 8.2 / max 13。

单轮 gpt 输出（4,157 轮实测）：动作轮 avg 634 / max 3,892，summary 轮 avg 2,329 / max 3,553——全部符合采集配置 `max_output_tokens: 4096`（单轮口径与名义超限的 tokenizer 差异说明见 run1 文档 §4.3）。

## 4. 训练注意

1. 本批任务偏长：总 tokens p90 ≈ 36k（run23 是 33.6k），`cutoff_len: 32768` 尾截断约 10–15%；49152 覆盖 ~97%（p99 57k）。
2. 其余（serving 对齐 jinja、空 think 注入等）同 run1 文档 §5。
