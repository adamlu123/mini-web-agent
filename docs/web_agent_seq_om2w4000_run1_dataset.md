# web_agent_seq_om2w4000_run1 数据集说明

> LlamaFactory ShareGPT 格式的 web-agent SFT 数据。
> 生成日期：2026-07-07（窗口开头修复版，commit `f4e79cb`；同日补充 self-reflection 负例）
>
> 文件（均已注册 `LlamaFactory/data/dataset_info.json`，注册名 = 文件名去 `.json`）：
>
> | 文件 | 大小 | 内容 |
> |---|---|---|
> | `web_agent_seq_om2w4000_run1.json` | 439M | 基础版 7774 条（self_reflection_final 全为正例） |
> | `web_agent_seq_om2w4000_run1_reflect0.json` | 12M | 补充的 1000 条 label=0 负例（见 §3.1） |
> | `web_agent_seq_om2w4000_run1_with_reflect0.json` | 448M | 上两者合并，8774 条，**训练推荐用这个**。文件内顺序 = trajectory_session（原顺序）+ 全部 aux 样本（image_qa / self_reflection_image / self_reflection_final 正负混合，seed=42 打乱，final 的 0/1 最长连续 8、平均 2）|

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

## 3. 样本构成（合并版 8774 条 = 基础版 7774 + 负例 1000）

| aux_type | 数量 | 说明 |
|---|---|---|
| trajectory_session | 5222 | 主体：窗口化的多轮 agent 会话（3730 带 summary 目标 / 1492 不带） |
| self_reflection_image | 1000 | 判分工具的单图打分样本，目标是 1–5 的 Score（分布 1:119 / 2:394 / 3:229 / 4:223 / 5:35），本身覆盖好坏截图，无需补负例（**触默认上限 1000 被截断**） |
| self_reflection_final | 2000 | 判分工具的最终聚合判定样本，`Status: success` / `Status: failure` **各 1000**（基础版正例触上限 1000 截断 + §3.1 负例 1000） |
| image_qa | 552 | 截图问答工具样本（未触上限） |

- 基础版图片引用共 6322 张（2552/7774 条样本带图）；负例另引用 4771 张（平均每条 4.8）；`images` 列已注册。
- 如需全量工具正例：重跑基础版加 `--max-self-reflection-image 0 --max-self-reflection-final 0`。

### 3.1 self-reflection 负例补充（reflect0）

**背景**：基础版走 `--require-self-judge-label1`，self_reflection_final 1000 条全为 success——模型只见过"判成功"，判定容易全过。而源 run 全量 6430 个 `judge_result.json` 中 label=1 仅 1569 个（24.4%）；任务级 2082 个任务中 1524 个至少一次成功 / 558 个全败。

**采样规则**：每个任务的 `final_runs/run_*` 按编号**从后往前**扫，取第一个判 failure 的 run（若最后一个 run 是 label=1，即落到其前最近的 label=0 run），每任务最多 1 条。扫描 1502 个任务凑满 1000 条（其中 502 个任务无失败 run 可取）；1000 条覆盖 1000 个不同任务，其中 558 个与正例同任务（不同 run）、与正例 source 零重复。

**构造**：复用 `make_web_agent_aux_sft.py::_self_reflection_examples_from_result`（与正例同一代码路径），只保留 `aux_type == self_reflection_final`。脚本：`LlamaFactory/scripts/make_web_agent_reflect0_sft.py`。

**已校验**：1000 条全部以 `Status: failure` 结尾；`<image>` 占位符与 images 一一对应；图片路径全部存在；字段与正例完全一致。

**判 0 的口径（坑）**：这批 `judge_result.json` 的 `predicted_label` 在 failure 时存的是 **null 而非 0**（解析未落 0），过滤/统计必须用 `final_response` 里的 `Status: failure` 文本重新解析（全量 4861 个 null 中 4860 个是 failure，1 个 response 未写 Status）。**生成的 SFT 数据里已规范化**：负例样本的 `predicted_label` 字段统一写 0（该字段仅溯源用，训练只读 conversations/system/images）；合并版中 `predicted_label` 分布 = image 1000 个 1 + final 各 1000 个 1/0。

## 4. 统计（Qwen3.5-9B tokenizer；tokens = system+全部轮次文本，不含 vision token；output = gpt 轮）

> 本节统计基于基础版 7774 条；负例是短的两轮判定样本（与正例同形态），对整体分布影响很小，未重算。

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

0. **数据集选择**：训练 yaml 里 dataset 用 `web_agent_seq_om2w4000_run1_with_reflect0`（final 判定正负均衡）；基础版 `web_agent_seq_om2w4000_run1` 的 self_reflection_final 全是正例，只判 success，不要单独用于训练判定能力。
1. **cutoff_len**：trajectory_session 的 p90 ≈ 33k，`cutoff_len: 32768` 会尾截断约 10% 主体样本（p99 47k、max 74k）。要基本无截断：49152 覆盖 ~99%，65536 近乎全覆盖。
2. **serving 对齐**：本数据 assistant 历史轮含完整 `<think>`，训练后 ckpt 推理必须挂 `configs/qwen3_5_train_aligned.jinja`，详见 `docs/qwen3_5_think_alignment.md`（坑 A/坑 B、MARKER 探针）。
3. 无 `<think>` 的工具类 target（self_reflection/image_qa）训练时会被 LlamaFactory 注入空 think 前缀并计 loss（坑 B），评分侧需归一化。
4. dataset_info 条目使用 `images` 列；带图样本走 qwen3_5 的 mm 通路，注意 `image_max_pixels` 与采集时（262144）一致。

## 6. 复现命令

```bash
cd /data/t-yifeili/mini-web-agent

# 基础版（7774 条）
/data/t-yifeili/miniconda3/envs/echo-rl/bin/python \
  LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py \
  --src /data/t-yifeili/om2w_4000_sampling/outputs/om2w_4000_run1 \
  --out LlamaFactory/data/web_agent_seq_om2w4000_run1.json

# self-reflection 负例（1000 条）
cd LlamaFactory/scripts && /data/t-yifeili/miniconda3/envs/echo-rl/bin/python \
  make_web_agent_reflect0_sft.py \
  --src /data/t-yifeili/om2w_4000_sampling/outputs/om2w_4000_run1 \
  --out ../data/web_agent_seq_om2w4000_run1_reflect0.json \
  --max-examples 1000

# 合并版 = 基础版 + 负例，直接 list 拼接后 dump 为
# LlamaFactory/data/web_agent_seq_om2w4000_run1_with_reflect0.json
```
