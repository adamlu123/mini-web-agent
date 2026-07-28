# web_agent_seq_om2w4000_run23 轨迹数据说明

> run1 的增量批：run2+run3 新采集轨迹，只取 trajectory_session（无工具样本）。
> 生成日期：2026-07-13，转换脚本与参数和 run1 完全一致（`make_web_agent_sequential_compact_tools_sft.py` 默认参数，prompt-mode `sft_state_debug`，label=1 过滤）。格式细节见 `docs/web_agent_seq_om2w4000_run1_dataset.md`。

## 1. 数据集与路径

| 数据集 | 条数 | 本机路径 | 集群路径（PVC） |
|---|---|---|---|
| run23 纯轨迹 | 2125 | `LlamaFactory/data/web_agent_seq_om2w4000_run23_traj_only.json`（174M，无图片） | `/mnt/pvc/t-yifeili/data/web_agent_seq_om2w4000_run23_traj_only_portable_bundle` |
| **合并版（训练推荐）** | **10899** | `LlamaFactory/data/web_agent_seq_om2w4000_run1r0_plus_run23traj_portable_bundle/`（5.9G） | `/mnt/pvc/t-yifeili/data/web_agent_seq_om2w4000_run1r0_plus_run23traj_portable_bundle` |

> **注**：本文的合并版已被 run4 增量后的 `web_agent_seq_om2w4000_run1r0_plus_run234traj_portable`（11403 条）取代，见 `docs/web_agent_seq_om2w4000_run4_dataset.md`。

- **合并版** = run1 with_reflect0 全量 8774（含工具样本、判定正负各 1000、全部图片）+ run23 纯轨迹 2125。图片 9825 张 / 引用 11093 处 / missing 0，注册名 `web_agent_seq_om2w4000_run1r0_plus_run23traj_portable`（bundle 内自带 dataset_info.json），训练 yaml 的 `dataset_dir`/`media_dir` 都指 bundle 根即可。
- `/mnt/pvc/t-yifeili/` 与 `/mnt/pvc/experiments/t-yifeili/` 是同一目录的两个入口。

## 2. 来源与过滤

- 采集：run2（2026-07-08，完成 172）+ run3（2026-07-13，905 全出结果：507 Submitted / 180 step 耗尽 / 218 RateLimitError）。模型 gpt-5.4 + browserbase，配置与 run1 相同。
- 任务筛选（856 个候选）：run2 取有 result.json 的 172；run3 排除 218 条 RateLimitError（**这些也有 result.json，判完成必须看 batch.log 的 exit_status**）取 687；再排除 3 个 run1 已用过 trajectory 的任务 → 与 run1 **零任务重叠**。
- label=1 过滤后 **642 个任务**进入数据集，切成 2125 条窗口样本（1487 带 summary 目标 / 638 不带）。
- 已校验：window-0（642 条）全部以 `Task:` 开头，后续窗口全部以 `## Compacted History Summary` 开头。

## 3. 统计（Qwen3.5-9B tokenizer，口径同 run1 文档 §4；2125 条 trajectory_session）

整条轨迹（642 条）：实际步数 avg 29 / p90 59 / max 102；训练 gpt 轮 avg 27 / p90 54 / max 97。

每条 training example（**output 为 example 内全部 gpt 轮加总**，平均 8.2 轮，非单轮）：

| 指标 | avg | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| example 总 tokens（全轮加总） | 21,759 | 21,298 | 27,520 | 33,634 | 50,593 | 73,761 |
| example output tokens（gpt 轮加总） | 6,479 | 6,987 | 8,868 | 10,407 | 15,073 | 31,866 |

每条 example 的 gpt 轮数 avg 8.2 / max 12。分布与 run1 主体基本一致。

单轮 gpt 输出（17,502 轮实测）：动作轮 avg 646 / max 4,116，summary 轮 avg 2,303 / max 3,953——全部符合 `max_output_tokens: 4096`（4,116 系 Qwen 与 gpt-5.4 tokenizer 计数差异，见 run1 文档 §4.3）。

## 4. 训练注意

1. `cutoff_len: 32768` 尾截断约 10%（p90 ≈ 33.6k）；49152 覆盖 ~99%。
2. 本批 output max 31.9k（run1 是 23.7k），个别 gpt 轮更长。
3. 其余（serving 对齐 jinja、空 think 注入等）同 run1 文档 §5。
