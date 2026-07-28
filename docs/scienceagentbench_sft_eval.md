# ScienceAgentBench eval for science-SFT checkpoints

集成 ScienceAgentBench（SAB）到 miniswewebagent harness，用于对 D3-Gym 科学 SFT
模型（`d3gym_science_sft_*`）做 SFT 后评测。可本地单机跑，也可上集群（推理阶段）。

## 1. SAB 原生流程分析（`/data/t-yifeili/ScienceAgentBench`）

SAB 是**单轮 / 自调试代码生成 + 执行评测** benchmark，102 个 verified 任务，四个科学领域。

- **数据**：HF `osunlp/ScienceAgentBench`（split=`verified`，102 条，只含 *输入*：
  `task_inst` / `dataset_folder_tree` / `dataset_preview` / `domain_knowledge` /
  `gold_program_name` / `output_fname` / `eval_script_name`）。完整 benchmark（含
  `benchmark/datasets`、`gold_programs`、`eval_programs`）需另外下载解压，已在本机
  `ScienceAgentBench/benchmark/`（datasets 3.8G / gold 102 / eval 111）。
- **推理**（`run_infer.py` + `agent.py::ScienceAgent`）：每任务给一个 system prompt
  （"expert Python programming assistant"）+ 任务文本（含 `benchmark/datasets/<repo>`
  路径树 + 预览），模型输出一个 ```python``` 代码块 → 写成 `pred_programs/pred_<gold>.py`。
  `--use_self_debug` 时最多 10 轮：装依赖(pipreqs+pip-compile)→跑→把报错回灌让模型改。
  产物脚本必须把结果存到任务指定的 `pred_results/<output_fname>`。
- **评测**（两种）：
  - 容器化（推荐，`python -m evaluation.harness.run_evaluation`，需 docker）：每实例
    起容器装环境、跑 `pred_<gold>.py`、执行 `eval_programs/<eval_script>` 打分，8 线程
    ~30 分钟一遍。
  - 直接（`run_eval.py`，需 `sci-agent-eval` conda 环境 + pip-tools + OPENAI_API_KEY
    给可视化 judge）。
  - 指标（`calculate_metrics.py`）：读 run log + eval log，多次 run 取 best，报
    **Success Rate / CodeBERTScore / Valid Program Rate / Cost**。

**关键点**：SAB 的评分是**独立于策略模型**的——只吃 `pred_programs/` 里的脚本。所以
我们只需让 SFT 模型产出高质量 `final_script.py`，再喂给 SAB 原生评测即可。

## 2. 集成设计

复用现有 om2w batch harness（`miniswewebagent.run.benchmarks.om2w`）跑 SAB 任务，把
每个任务的 `final_script.py` 收集成 SAB 的 `pred_programs/`，再交给 SAB 原生评测。
Harness 里**不做**科学代码的正确性判定（无 self_reflection、无 om2w web judge）。

SFT 模型是多轮 sft_state agent（D3 数据训练），所以每个 SAB 任务当成一个多轮会话跑
（探查→写 `final_script.py`→运行→done），与 D3 SFT 训练分布对齐；SAB 原生的单轮/
self-debug 循环不用。

新增/改动文件：

| 文件 | 作用 |
|---|---|
| `scripts/make_sab_tasks_json.py` | HF verified → `sab_verified.json`（102 任务）。任务文本按 D3 格式拼：`task_inst` + `benchmark/datasets/` 根的 `├──/└──` 树 + previews。`--use-knowledge` 可选附专家知识（默认关，贴近 D3 训练分布）。 |
| `src/miniswewebagent/run/benchmarks/sab_verified.json` | 生成的任务文件（level 全为 `sab`；`sab` 字段存 gold_program_name/output_fname/eval_script_name/dataset_repo）。 |
| `src/miniswewebagent/config/benchmark/sab_science_sft_vllm.yaml` | SAB eval 配置：openai_compatible + `response_mode: sft_state`（→ `parse_sft_state_output`）；system/instance 模板 = D3 科学 prompt（与 `LlamaFactory/scripts/make_d3gym_science_sft.py` 常量一致）；observation 模板与 D3 SFT 数据逐字一致；`require_self_reflection_success: false`、`run.judge_enabled: false`；`environment.seed_symlinks` 把 SAB `benchmark/datasets` 软链进每个任务 workspace。**`agent.prompt_mode` 故意不设**——一旦设为 sft*/sft_state，`run.mini._apply_prompt_mode` 会用 echo_rl 的浏览器 SFT prompt 覆盖掉这里的科学模板。 |
| `src/miniswewebagent/environments/local_workspace.py` | 新增 `seed_symlinks: dict[str,str]`，`prepare()` 时在 workspace 内建软链（父目录自动创建，已存在则跳过）。不影响现有 web eval。 |
| `scripts/collect_sab_preds.py` | batch 输出目录里每任务 `final_script.py` → `pred_programs/pred_<gold>.py`（缺失写 `ERROR` 占位）；按 SAB verified 顺序生成 run log jsonl（cost=0）。 |
| `scripts/sab_eval_sft_vllm.sh` | 一键 Stage A：vllm serve ckpt → om2w batch 跑 SAB 任务 → collect_sab_preds → 打印 Stage B（SAB 评测）命令。 |

## 3. 本地用法

```bash
cd /data/t-yifeili/mini-web-agent

# (一次) 生成任务文件；改数据/加知识时重跑
HF_HOME=/data/t-yifeili/hf_cache \
  /data/t-yifeili/miniconda3/envs/echo-rl/bin/python scripts/make_sab_tasks_json.py

# Stage A：推理（起 vllm + 跑 102 任务 + 收集 pred_programs）
CKPT=/path/to/qwen35_9b/full/d3gym_32b MODEL_NAME=d3gym_32b TP=4 WORKERS=8 \
  bash scripts/sab_eval_sft_vllm.sh
#   SMOKE=1 只跑前 2 个任务；LIMIT=N 跑前 N；START_VLLM=0 复用已起的 endpoint
#   SAB_REPO=/path 指非默认 ScienceAgentBench checkout

# Stage B：SAB 原生评测（脚本结束会打印带实际路径的完整命令），在 SAB 仓库跑：
cd /data/t-yifeili/ScienceAgentBench
export OPENAI_API_KEY=<key>            # 可视化 judge 需要
python -m evaluation.harness.run_evaluation \
    --benchmark_path benchmark \
    --pred_program_path <PRED_OUT> \
    --log_fname d3gym_32b_eval.jsonl --run_id d3gym_32b
python calculate_metrics.py --run_logs <RUN_LOG> --eval_logs d3gym_32b_eval.jsonl
```

## 4. 集群

- **Stage A（推理）可上集群**：与现有 web-agent SFT eval 同套路（vllm serve + harness
  batch）。可套 `.claude/skills/web-agent-dist-train-eval` 的多节点/分片模式（om2w runner
  支持 `--num-shards/--shard-index`），也可训练完同 job 追加。集群节点需能读 SAB
  `benchmark/datasets`（用 `SAB_BENCHMARK_DATASETS=` 或 `-c environment.seed_symlinks...`
  指到 PVC 上的 SAB 数据副本）。
- **Stage B（评测）建议本机跑**：SAB 容器化评测需 docker + 逐任务建 conda 环境
  （numpy/torch/rdkit/... 各版本），本机 docker 可用。集群 pod 一般没有 docker-in-docker，
  跑不了容器化评测；且 Stage B 只吃 `pred_programs/`（几百 KB 脚本），从集群拉回本机再
  评测最省事。

## 5. Smoke 验证（2026-07-13，用 web SFT ckpt `eval_st_1ep` 验管路）

用非科学的 web SFT ckpt 跑通 Stage A 全链路（只验管路，脚本质量无意义）：vllm serve ✓、
配置合并 + level=sab 选到 102 任务 ✓、workspace 内 `benchmark/datasets` 软链解析到 SAB
数据 ✓、科学 prompt 下发 + sft_state 解析 + 多轮循环 ✓、模型写出时 `final_script.py`
被捕获 ✓、collector → `pred_programs/pred_<gold>.py` + 缺失 ERROR 占位 + run log ✓。

**已修 bug**：初版 `max_output_tokens=12000` + `max_model_len=36864` + 观察截断 20000，
科学任务大输出累积到第 9 步撑爆上下文（vLLM 400）。已收紧：`max_output_tokens=8000`、
观察截断 10000、`summary_every_n_steps=6`（更早压缩）、驱动脚本 `MAX_MODEL_LEN=40960`。
复跑无 400。

## 6. 注意 / 坑

- **别设 `agent.prompt_mode`**（见上，会覆盖科学模板）。解析靠 `model.response_mode: sft_state`。
- 完结门只在 prompt 层（无硬性 "final_script.py 必须存在" 门）；SFT 模型靠训练遵守，
  过早 done 的任务 pred 会是 ERROR，SAB 记 0 分——不影响其余任务。
- 任务文本默认不含 domain_knowledge（对齐 D3 训练分布）；想对齐 SAB 论文 with-knowledge
  设定，`make_sab_tasks_json.py --use-knowledge` 重生成。
- Stage B 的可视化 judge（判 .png 输出类任务）需 `OPENAI_API_KEY`。
