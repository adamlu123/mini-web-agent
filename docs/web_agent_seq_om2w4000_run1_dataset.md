# web_agent_seq_om2w4000_run1 数据集说明

> LlamaFactory ShareGPT 格式的 web-agent SFT 数据。
> 文件：`LlamaFactory/data/web_agent_seq_om2w4000_run1.json`（439M，git-lfs）
> 注册名：`web_agent_seq_om2w4000_run1`（`LlamaFactory/data/dataset_info.json`）
> 生成日期：2026-07-07（窗口开头修复版，commit `f4e79cb`）

## 1. 数据来源

| 项 | 值 |
|---|---|
| 采集 run | `/data/t-yifeili/om2w_4000_sampling/outputs/om2w_4000_run1` |
| 任务集 | `om2w_4000_instructions.json`（实际 3149 条：medium 1311 / easy 983 / hard 855，mind2web_expanded 来源） |
| 策略模型 | gpt-5.4（phyagi gateway），browserbase 后端，`best_default_judge_json_agnostic.yaml` 变体 |
| agent 配置要点 | step_limit=100，`summary_every_n_steps` 触发 in-place 历史压缩（compaction） |
| 采集结果 | 2072 条完整结束（1507 Submitted / 564 step 耗尽 / 1 content_filter），另 ~100 条中途手动停止 |

## 2. 过滤与转换

- 转换脚本：`LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py`（默认参数，`--window-size 20`，prompt-mode `sft_state_debug`）
- 过滤：`--require-self-judge-label1`（默认开）。扫描 2172 个 `trajectory.json`，**1524 条**存在 `final_runs/run_*/judge_result.json` 且 `predicted_label==1`，其余 648 条丢弃。
  （1524 > 1507 Submitted：少数超步/中断任务曾在中途某个 run 拿到过 label=1。）
- 目标格式：每个 assistant 轮 `<think>…</think>\n<bash>…</bash>\n<done>…</done>\n<final_response>…</final_response>`；操作者路径归一化为 `/workspace`，密钥脱敏。

### 2.1 窗口切分（sequential compact）

每条轨迹按 agent 的 compaction 边界切成若干 example：

- **window 0**（turn 1~首次压缩）：以**原始任务 prompt** 开头（`Task: ...\nTask ID: ...\nStart URL: ...`）；
- **后续窗口**：以 `## Compacted History Summary`（上一窗口的压缩摘要）开头；
- 有摘要目标的窗口，最后一轮 gpt 是 compact 摘要本身（`has_summary_target=true`，5222 条中 3730 条）。

**已修复的坑（commit `f4e79cb`）**：本批轨迹的 `trajectory.json` messages 被 agent 的 in-place compaction 重写，旧脚本 `_initial_user` 读到的"首条 user"已是压缩摘要，导致 window-0 错误地以 `## Compacted History Summary` 开头。修复后 window-0 优先从 `debug/requests/request_0001.json`（step-1 真实请求 payload）取原始 prompt。凡轨迹带 compaction 的数据源，必须用含 `_initial_user_from_requests` 的脚本版本。

## 3. 样本构成（共 7774 条）

| aux_type | 数量 | 说明 |
|---|---|---|
| trajectory_session | 5222 | 主体：窗口化的多轮 agent 会话（3730 带 summary 目标 / 1492 不带） |
| self_reflection_image | 1000 | 判分工具的单图打分样本（**触默认上限 1000 被截断**） |
| self_reflection_final | 1000 | 判分工具的最终聚合判定样本（**触默认上限 1000 被截断**） |
| image_qa | 552 | 截图问答工具样本（未触上限） |

- 图片引用共 6322 张（2552/7774 条样本带图）；`images` 列已注册。
- 如需全量工具样本：重跑加 `--max-self-reflection-image 0 --max-self-reflection-final 0`。

## 4. 统计（Qwen3.5-9B tokenizer；tokens = system+全部轮次文本，不含 vision token；output = gpt 轮）

### 4.1 整条轨迹的 turn 数（1524 条轨迹，窗口拼接至任务结束）

| 口径 | avg | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| agent 实际步数（n_calls，末窗口 end_call） | 31 | 23 | 40 | 63 | 99 | 102 |
| 训练样本 gpt 轮数（含 compact 摘要轮） | 29 | 22 | 37 | 58 | 90 | 98 |

（两口径之差：n_calls 编号包含 compact/格式重试调用，它们不产生训练轮。）

### 4.2 每条 training example 的 token 长度

全部 7774 条：

| 指标 | avg | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 总 tokens | 14,660 | 15,628 | 23,740 | 30,294 | 44,276 | 74,168 |
| output tokens | 4,374 | 4,134 | 7,729 | 9,508 | 13,344 | 23,693 |

主体 trajectory_session 5222 条：

| 指标 | avg | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 总 tokens | 21,280 | 21,056 | 26,760 | 32,927 | 46,757 | 74,168 |
| output tokens | 6,351 | 6,820 | 8,638 | 10,166 | 13,876 | 23,693 |

按 example 的 gpt 轮数：全体 avg 5.9 / max 19；trajectory_session avg 8.3 / max 19。

## 5. 训练注意事项

1. **cutoff_len**：trajectory_session 的 p90 ≈ 33k，`cutoff_len: 32768` 会尾截断约 10% 主体样本（p99 47k、max 74k）。要基本无截断：49152 覆盖 ~99%，65536 近乎全覆盖。
2. **serving 对齐**：本数据 assistant 历史轮含完整 `<think>`，训练后 ckpt 推理必须挂 `configs/qwen3_5_train_aligned.jinja`，详见 `docs/qwen3_5_think_alignment.md`（坑 A/坑 B、MARKER 探针）。
3. 无 `<think>` 的工具类 target（self_reflection/image_qa）训练时会被 LlamaFactory 注入空 think 前缀并计 loss（坑 B），评分侧需归一化。
4. dataset_info 条目使用 `images` 列；带图样本走 qwen3_5 的 mm 通路，注意 `image_max_pixels` 与采集时（262144）一致。

## 6. 复现命令

```bash
cd /data/t-yifeili/mini-web-agent
/data/t-yifeili/miniconda3/envs/echo-rl/bin/python \
  LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py \
  --src /data/t-yifeili/om2w_4000_sampling/outputs/om2w_4000_run1 \
  --out LlamaFactory/data/web_agent_seq_om2w4000_run1.json
```
