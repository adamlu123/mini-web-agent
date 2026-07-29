# SPB 谱系完整链路：训练 / Eval / WebJudge（2026-07-29 现行版）

本文描述当前正在使用的全部三条链路的**每一步、每个文件、每条命令**。
（暂未合入 SKILL.md，供审阅。）

---

## 一、训练链路（数据 → PVC → train+convert 同 job）

### 1.1 数据构造（本机）

**输入**：采集轨迹目录（每任务一个 `m2w_exp_*/`，含 `trajectory.json` / `result.json` / `raw_responses.jsonl` / `debug/steps/*.json`）。

**选择口径**：`exit_status == "Submitted"` 且 首个 `self_reflection --scope trajectory` 观察为 `Status: ok`（"首过"）。

**system prompt**：SPB 冻结模板 `mini-web-agent/echo_rl/web_agent/sft_assets/spb_state_system.txt`
（= 采集版 JSON prompt 的 XML 换装，与 luyadong 0720 bundle 逐字同源，仅修了 `<<final_response>>` 转换笔误），按任务 `{{ start_url }}` jinja 渲染。

**gpt 轮重建**：assistant 消息的 `extra.raw_response`（thought/bash_command/done/final_response）
→ `<think>…</think>\n<bash>…</bash>\n<done>…</done>\n<final_response>…</final_response>`。

**生成器**（`/data/t-yifeili/0723/260718_om2w4000_first1000_xhigh_4x/training_bundles/singleturn_tools/`）：

| 脚本 | setting | 粒度/mask |
|---|---|---|
| `gen_sw_settings.py --runs run_2 --tag run2_fp570_spb` | sw10（滑窗10）+ full24k（全历史） | 单轮 / mask=last |
| `gen_lastobs.py` | lastobs（历史think+bash+最新obs）+ lastobs_think（历史只留think） | 单轮 / mask=last |
| `gen_sum10.py --granularity mt` | sum10（每10步compaction，窗口+summary目标） | **多轮窗口 / mask=all** |

```bash
cd /data/t-yifeili/0723/260718_om2w4000_first1000_xhigh_4x/training_bundles/singleturn_tools
/data/t-yifeili/miniconda3/envs/echo-rl/bin/python gen_lastobs.py            # 举例
```

**产物 bundle**：`training_bundles/<name>/`，含 `om2w_<name>.json`（ShareGPT 行：system/conversations/images/元数据）
+ `dataset_info.json` + `manifest.json` + `source_tasks.json`。

### 1.2 上传 PVC

```bash
UPLOAD_IMAGE=ubuntu:24.04 bash /data/t-yifeili/mini-web-agent/docker/upload_data_to_pvc.sh \
  /data/t-yifeili/0723/260718_om2w4000_first1000_xhigh_4x/training_bundles/<bundle名>
# → /mnt/pvc/experiments/t-yifeili/data/<bundle名>（分块 kubectl cp + 字节校验）
```

### 1.3 提交 combo（train → convert → eval 同一节点不释放）

**提交器**：`/data/t-yifeili/aifsdk/phitrain/recipes/train/sft_qwen3.5/scripts/send_job_ctx2_combo.sh`

```bash
cd /data/t-yifeili/aifsdk/phitrain && source ~/.bashrc
VARIANT=spblastobs bash recipes/train/sft_qwen3.5/scripts/send_job_ctx2_combo.sh
# VARIANT ∈ spblastobs|spbsw10|spblastobsth|spbfull24k|spbsum10（sum10 需 MAX_STEPS=3*rows/32）
# 默认 p0 / PRIORITY_CLASS_NAME=high / workstream=agenticbrain-sft / 1节点8卡 / wandb.ai
```

VARIANT 决定：bundle 路径、`ASSISTANT_MASK_MODE`（单轮=last，多轮=all）、
tokenize seq（32768；sum10=49152 配 `9b-48k.yaml`+`webwright-48k.yaml`）、eval 配置。

**job 内执行体**：`aifsdk/phitrain/scripts/train_convert_eval_ctx2.sh`（三阶段全部幂等可续）：

```text
Phase 1 train:
  python -m scripts.tools.data.tokenization.tokenize_webwright_vlm_sft \
    --bundle-dir $BUNDLE --max-seq-len $SEQ --assistant-mask-mode $MASK \
    --chat-template .../Qwen3.5-no-auto-think/chat_template.jinja
    # 注意:超长行整行丢弃(不截断)
  torchrun scripts/train/train.py 9b.yaml sft-qwen3.5-9b-webwright.yaml webwright-32k.yaml \
    --training_args.trainer.max_steps $STEPS \        # 823≈单轮3epoch;sum10 动态
    --training_args.save_steps 0 --save_top_k 0 --save_final_checkpoint true  # 只存末尾一份(防PVC爆额)
Phase 2 convert:
  phitrain-cli convert train/last -d bfloat16 -t Qwen/Qwen3.5-9B   # distcp→HF text
  cp -rl 导出 → models/ctx2-<variant>-hf-vlm
  merge_vision_from_base_ctx2.py --ckpt … --base <Qwen3.5-9B snapshot>  # 补 vision 权重
  再拷 base 的 config.json/processor*/chat_template.jinja             # vLLM 要完整 VL config
Phase 3 eval: 见下一节(同节点直接执行 run_dist_eval_q35_image.sh)
```

---

## 二、Eval 链路（生成轨迹）

### 2.1 两个入口

- **combo Phase 3**（自动，紧接训练）
- **standalone**（已有 HF ckpt 单独评）：

```bash
cd /data/t-yifeili/mini-web-agent && source ~/.bashrc
export PROJECT_NAME=agenticbrain-sft PRIORITY=p0 PRIORITY_CLASS_NAME=high
JOB_NAME=t-yifeili-p0-absft-ev-<名> \
EVAL_CKPT=/mnt/pvc/experiments/t-yifeili/models/ctx2-<variant>-hf-vlm \
NODES=1 TOTAL_WORKERS=16 \
BENCHMARK_CONFIG=benchmark/om2w_spb_vllm_<variant>.yaml \
MAX_MODEL_LEN=32768 MAX_OUTPUT_TOKENS=4096 \
MAX_CONTEXT_TOKENS=28672 SLIDING_WINDOW_KEEP_TURNS=999 \
EVAL_RUN_ID=ctx2_<variant>_all \
bash docker/submit_dist_eval_q35_image.sh
# 重提同 EVAL_RUN_ID = 断点续评;RETRY_FAILED=1 重跑失败任务
```

### 2.2 job 内发生什么（`docker/run_dist_eval_q35_image.sh`）

1. 上传的 mini-web-agent 拷到 pod 本地 + 软链 `/home/luyadong/sandbox/mini-web-agent`（vendored judge 路径约定）；
2. 依赖 bootstrap（不动镜像 CUDA/torch/vLLM 栈）；
3. 无 manifest → manifestless 兼容分支：chat template 默认 `configs/qwen3_5_train_aligned.jinja`（与 phitrain 训练模板对齐）；
4. `vllm serve $EVAL_CKPT --tp 8 --max-model-len 32768`（combo 里追加 `--disable-custom-all-reduce`）；
5. `python -m miniswewebagent.run.benchmarks.om2w -c <BENCHMARK_CONFIG> …`：
   - **上下文构造 = 各组 yaml**（`src/miniswewebagent/config/benchmark/om2w_spb_vllm_*.yaml`）：
     内联 SPB system/instance（与训练 bundle 逐字一致）、`output_truncation_chars: 24000`、
     `step_limit: 50`、各组开关（`context_window_steps: 10` / `history_context_mode: last_obs|last_obs_think` /
     `summary_every_n_steps: 10`+定制 summary_user_prompt）；
   - `-c` 覆盖：endpoint/model 名、`model.max_context_tokens=28672`+`sliding_window_keep_turns=999`
     （= 模型层 token 驱逐等效禁用；超 32k 请求原样发 vLLM → 400 → episode 终止，"超窗即停"）；
   - 逐任务产物 → `/mnt/pvc/experiments/t-yifeili/evals/<EVAL_RUN_ID>/outputs/<task_id>/`
     （task.json / steps/step_N.sh / screenshots/*.png / result.json / trajectory.json / debug/requests/）；
6. 末尾自带 sandbox judge（**仅对照用**，正式分见第三节）。

---

## 三、WebJudge 正式判分（原版 OM2W）

### 3.1 脚本与输入组装

**脚本**：`mini-web-agent/scripts/eval_persistent_cli_steps_with_original_om2w.py`
（main 分支版；包装 `eval_persistent_cli_with_original_om2w.py`，
vendored 上游 WebJudge 从 `/home/luyadong/sandbox/mini-web-agent/om2w_judge` 加载）。

**每任务的 judge 输入组装**（steps 布局，自动检测）：

| 成分 | 来源 | 规则 |
|---|---|---|
| 任务描述 | `<task_id>/task.json` | `task` / `confirmed_task` 字段 |
| action history | `<task_id>/steps/step_<id>.sh` | **每个非空脚本全文**为一条 action，按文件名数字排序 |
| 截图 | `<task_id>/screenshots/*.png` | 全部 PNG，按文件名尾号数字排序 |
| 判分 | 上游 `WebJudge_Online_Mind2Web_eval` | o4-mini、score_threshold=3：先逐图打分，再综合 action 史出 `predicted_label` ∈ {1,0} |
| 凭证 | `cred.sh` | `OPENAI_GATEWAY_ENDPOINT`+`OPENAI_GATEWAY_API_KEY`（自动 gateway 模式）；无则 direct OpenAI |

**输出**：`$OUT_DIR/WebJudge_Online_Mind2Web_eval_o4-mini_score_threshold_3_auto_eval_results.json`
（逐任务 JSONL；**断点续判**：重跑跳过已判 task_id）+ stdout 汇总
（`successful_tasks` / `success_rate_completed` / `missing_tasks`）。

### 3.2 集群运行命令（现行标准）

**runner**：`mini-web-agent/docker/run_pcli_judge.sh`（软链 + source cred + pip install backoff + 调 3.1 脚本）。

```bash
STAGING=$(mktemp -d) && mkdir -p $STAGING/mini-web-agent && \
for rel in src scripts docker om2w_judge om2w_judge_sandbox echo_rl configs pyproject.toml README.md LICENSE; do
  cp -a /data/t-yifeili/mini-web-agent/$rel $STAGING/mini-web-agent/ 2>/dev/null; done
kubectl -n bonete61 create secret generic t-yifeili-webchain-sampling-creds \
  --from-file=cred.sh=/data/t-yifeili/webchain_sampling/cred.sh --dry-run=client -o yaml | kubectl -n bonete61 apply -f -
export PROJECT_NAME=agenticbrain-sft PRIORITY=p0 PRIORITY_CLASS_NAME=high USER_ALIAS=t-yifeili
JOB_NAME=t-yifeili-p0-absft-pclijudge-<名> \
bash /data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh \
  --upload $STAGING/mini-web-agent \
  --image aifrontiers.azurecr.io/nvidia-26.06-pytorch-2.12.1-…-vllm-0.24.0:20260707 \
  --acr --node 1 --gpu-per-node 0 --cpu 32 --memory 64Gi \
  --secret-volume "t-yifeili-webchain-sampling-creds:/run/secrets/webchain-sampling" \
  --extra-env-vars "TRAJ_DIR=/mnt/pvc/experiments/t-yifeili/evals/<run_id>/outputs,OUT_DIR=/mnt/pvc/experiments/t-yifeili/evals/<run_id>/judge_pcli,NUM_WORKER=32,EXPECTED_TASKS=300" \
  --cmd 'bash $DATA_ROOT/runs/$JOB_NAME/mini-web-agent/docker/run_pcli_judge.sh'
```

**本地等价命令**（轨迹在本机可见时）：

```bash
cd /data/t-yifeili/mini-web-agent && source /data/t-yifeili/webchain_sampling/cred.sh
python scripts/eval_persistent_cli_steps_with_original_om2w.py \
  --trajectories_dir <outputs目录> --output_path <判分输出目录> \
  --model o4-mini --score_threshold 3 --num_worker 32 --expected_tasks 300
```

### 3.3 已知坑

- 镜像缺 `backoff` → runner 已内置 `pip install backoff`；
- 网关**团队日预算 $4000**：429 `budget.exceeded` 时判分全挂，等日重置后重提（增量续判无损）；
- 每次原版判分 300 任务 ≈ 数百刀 o4-mini 开销，一天多轮判分会撞预算。

---

## 附：当前记分板（SPB 谱系，原版 WebJudge）

| 组 | 分数 | 状态 |
|---|---|---|
| spblastobsth（历史只留 think + 最新 obs） | **20.7%**（62/300） | ✅ |
| spbsw10（滑窗 10） | 15.3%（46/300） | ✅ |
| spblastobs（历史 think+bash + 最新 obs） | 待判 | 生成 299/300 就绪，等网关日预算重置后判分 |
| spbsum10 | — | 等数据生成完 → `launch_sum10.sh` 一键 |
