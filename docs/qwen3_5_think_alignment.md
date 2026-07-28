# Qwen3.5 SFT ckpt 的 `<think>` 对齐：正确 inference 流程与排错手册

> **TL;DR：凡是用 LlamaFactory `template: qwen3_5` + `mask_history: false` 训出来的多轮 ckpt，
> serving 时必须挂 `configs/qwen3_5_train_aligned.jinja`，否则多轮 prompt 与训练分布不一致，
> 模型行为会从第 2 轮开始逐轮劣化。** 起服务后先跑一次本文的 MARKER 探针再开始任何 eval。

首次定位记录：2026-07-06，m2w_exp_0280 100ep overfit 探针实验（commit `2f5c53c`、`6dd4284`）。

---

## 1. 两个坑是什么

### 坑 A：历史轮 `<think>` 被 serving 端剥掉（影响所有多轮推理）

| | 历史 assistant 轮的 `<think>...</think>` |
|---|---|
| **训练**（LlamaFactory `qwen3_5`，`mask_history: false`，`enable_thinking: true`） | **完整保留**在 prompt 里（`ReasoningTemplate.encode_multiturn`，仅 `mask_history and not preserve_thinking` 时才丢弃） |
| **推理**（vLLM + ckpt 自带 stock Qwen3.5 chat template） | **全部剥掉**（stock 模板只保留非最后一轮 assistant `</think>` 之后的内容），且 generation prompt 末尾替模型追加 `<think>` |

后果：turn 1（无历史）与训练完全一致，可精确复现；轮次越深，被剥的 think 越多，
输入越偏离训练分布，输出偏离越大。overfit 探针上的表现：turn 1 逐字节复现 gold，
turn 2 起在近似平局 token 处翻转，`main_agent_next_turn` 的 `bash_exact` 掉到 0.46。

**症状特征（以后见到这个模式先怀疑本坑）**：teacher-forced replay 中第 1 轮 exact、
后续轮次单调劣化；同一 prompt 的 `usage.prompt_tokens` 明显小于按训练渲染估算的值。

### 坑 B：无 `<think>` 的 target 会被训练端注入空 think（影响 judge/reflection/image_qa 类数据）

LlamaFactory `ReasoningTemplate` 对**不含 think 标签的 assistant target**（如
`self_reflection_image` / `self_reflection_final` / image_qa 这类直接以 `Reasoning:` /
`Thoughts:` 开头的 judge 输出），会在 tokenize 时自动前置
`"<think>\n\n</think>\n\n"` 并**计入 loss**（`template.py` 的 "add empty cot" 分支，
`enable_thinking: true` 时走"算 loss"路径）。

后果：模型对这类样本输出自带空 think 前缀——这是**忠实复现训练目标**，不是错误。
但数据 JSON 里的 gold 字符串没有这个前缀，所以：

- 任何"整段字符串精确比较"的评分都会误判为 mismatch；
- 下游解析器要能容忍输出开头的空 `<think>\n\n</think>\n\n`（按行/正则提取 `Status:` 、
  `Score:` 、标签字段的解析方式天然兼容）。

---

## 2. 正确的 inference 流程

### 2.1 起服务（必须）

```bash
vllm serve "$CKPT" \
  --served-model-name policy \
  ... \
  --chat-template /path/to/mini-web-agent/configs/qwen3_5_train_aligned.jinja \
  --mm-processor-kwargs '{"max_pixels":262144}'   # 可选，见 2.3
```

`configs/qwen3_5_train_aligned.jinja` 与 LlamaFactory 训练渲染**逐字节一致**：

- 每条消息渲染为 `<|im_start|>{role}\n{content}<|im_end|>\n`，assistant 历史**原样输出**（含 think）；
- generation prompt 只有 `<|im_start|>assistant\n`，**不追加** `<think>`（训练 target 以
  `<think>\n` 开头，模型自己生成开标签）;
- 图片 part 渲染为 `<|vision_start|><|image_pad|><|vision_end|>`，与 `qwen3_vl` mm_plugin 的
  `<image>` 占位一致。

一劳永逸的替代做法：`cp configs/qwen3_5_train_aligned.jinja $CKPT/chat_template.jinja`，
vLLM 会自动加载 ckpt 目录里的模板（`mini_harness_eval_sft_vllm.sh:51` 的 warn 就是在检查它），
之后 serve 该 ckpt 不再需要 `--chat-template`。二选一即可。

注意 `mini_harness_eval_sft_vllm.sh` 里 `${VLLM_ARGS:-}` 是**无引号展开**，通过 `VLLM_ARGS`
传参时 JSON 里不能有空格：`VLLM_ARGS='--chat-template ... --mm-processor-kwargs {"max_pixels":262144}'`。

### 2.2 起服务后验证（必须，5 秒）

```bash
curl -s http://127.0.0.1:${PORT}/tokenize -H 'Content-Type: application/json' -d '{
  "model":"<served-model-name>","add_generation_prompt":true,"return_token_strs":true,
  "messages":[{"role":"user","content":"hi"},
    {"role":"assistant","content":"<think>\nMARKER\n</think>\nans"},
    {"role":"user","content":"again"}]}' | grep -o MARKER
```

- 能 grep 到 `MARKER` → 历史 `<think>` 被保留，模板生效，可以开跑；
- grep 不到 → 还在用 stock 模板，**停下来先修**。

### 2.3 图片预处理对齐（可选）

训练配置 `image_max_pixels: 262144`，vLLM 默认分辨率上限更高，导致带图样本的 vision token
与训练略有差异（overfit 探针上表现为 8 个带图 case 中 1 个出现一对弯引号级别的偏差）。
追求逐字节对齐时加 `--mm-processor-kwargs '{"max_pixels":262144}'`；正常 eval 影响很小，可不加。

### 2.4 客户端注意事项

- 在线 harness（`sft_state` 模式）把历史 assistant 消息从 `raw_response` 重建为
  `<think>\n{thought}\n</think>\n<bash>...` 完整格式（`openrouter_model.py:_sft_state_assistant_content`），
  与本模板配合后渲染结果与训练一致，**客户端无需改动**。
- 输出解析 `_extract_sft_think_values`（`phyagi_model.py`）对带/不带 `<think>` 开标签的输出
  都兼容。**不要**在客户端给模型输出手动补 `<think>` 前缀（换模板后模型自己会输出，补了会双标签）。

---

## 3. 排错与复现工具

| 脚本 | 用途 |
|---|---|
| `replay_compare_turn.py <i>` | 单轮调试：取 `prompt_messages.jsonl` 第 i 行，打印完整输入/输出并与 debug bundle 对应 turn 做 diff。`--train-render` 按训练渲染手工拼 prompt 走 `/v1/completions`（绕过服务端模板），用于区分"模型没学会"和"输入不对齐"；`--no-send` 只比输入。 |
| `scripts/archive/sft_replay_all_cases_thinkfix.py` | 全量 teacher-forced replay + 评分。相比归档原版 `scripts/archive/sft_replay_all_cases.py`：①启动即跑 2.2 的探针，服务端剥 think 直接报错退出（`--skip-template-check` 跳过）；②评分做坑 B 归一化（gold 无 think 且 pred 带空 think 前缀时剥掉再比，记录 `stripped_injected_empty_think` 标记）；③输出 `prompt_token_counts.csv`。 |

复现命令（m2w_exp_0280 overfit 探针）：

```bash
.venv/bin/python scripts/archive/sft_replay_all_cases_thinkfix.py \
  --data m2w_exp_0280_debug_bundle/web_agent_debug_m2w_exp_0280.json \
  --media-dir <repo>/scripts \
  --out deterministic_replay_m2w_exp_0280_thinkfix --save-prompts
```

修复前后对比（100ep overfit ckpt，74 cases）：

| case_type | n | raw_exact | token_f1 | bash_exact |
|---|---|---|---|---|
| main_agent_next_turn | 54 | 0.00 → **1.00** | 0.707 → **1.000** | 0.463 → **1.000** |
| compact_summary | 6 | 0.00 → **1.00** | 0.789 → **1.000** | 1.0 → 1.0 |
| compact_to_main_agent | 6 | 0.00 → **1.00** | 0.993 → **1.000** | 1.0 → 1.0 |
| self_reflection_image | 7 | 0.00 → **0.857** | 0.958 → **0.996** | 1.0 → 1.0 |
| self_reflection_final | 1 | 0.00 → **1.00** | 0.999 → **1.000** | 1.0 → 1.0 |

（唯一不精确的 1 个带图 case 是 2.3 的图片分辨率问题，非 think 问题。）

---

## 4. 以后不再踩坑的 checklist

**训练侧（数据生成 pipeline）**

- [ ] 对无 `<think>` 的 assistant turn（self_reflection_image / self_reflection_final /
  image_qa 等 judge 类），生成数据时**显式**拼上前缀，使数据字符串 == 实际训练 target：

  ```python
  EMPTY_THINK = "<think>\n\n</think>\n\n"   # 必须与 thought_words 逐字节一致
  if "<think>" not in gpt_value and "</think>" not in gpt_value:
      gpt_value = EMPTY_THINK + gpt_value
  ```

  训练行为完全不变（LlamaFactory 检测到已有 think 就不再注入），但评分/调试从此无需特判。
- [ ] 保持 ShareGPT 多轮结构不变（本修复不要求 per-turn 拆样本）。
- [ ] 若改动 `template:` / `enable_thinking` / `mask_history` / `preserve_thinking` 中任何一项，
  重新核对训练渲染与 serving 模板是否仍一致（用 2.2 探针 + overfit 小样本 replay 验证）。

**推理/eval 侧**

- [ ] serve 该系列 ckpt 一律带 train-aligned 模板（`--chat-template` 或写入 ckpt 的
  `chat_template.jinja`），模板文件跟着 ckpt 走。
- [ ] 起服务后必跑 2.2 MARKER 探针。
- [ ] 跑 teacher-forced replay 用 `sft_replay_all_cases_thinkfix.py`（自带探针 + 归一化），
  不要再用旧版。
- [ ] 精确对齐实验加 `--mm-processor-kwargs '{"max_pixels":262144}'`。

**评分/解析侧**

- [ ] 整段字符串比较前，对"gold 无 think 而 pred 带空 think 前缀"的样本先归一化
  （或等数据侧 checklist 第 1 条落地后删掉此特判）。
- [ ] 判词解析用行/正则提取，不要假设输出第一行就是正文。
