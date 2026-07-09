---
name: web-agent-dist-train-eval
description: >-
  One-job "SFT train + multi-node OM2W harness eval" and standalone multi-node
  eval on the bonete61 B200 cluster, both resumable: warm-restart training from
  a checkpoint-N (RESUME_FROM_CKPT) and checkpointed eval (resubmit the same
  EVAL_RUN_ID to continue). Use when the user wants to: launch train+eval in a
  single job, continue a killed SFT run, run or resume a data-parallel eval of
  an HF ckpt across N nodes, retry failed eval tasks, or locate judge results
  (run_summary_judge.json, level breakdown). Data prep / PVC data upload /
  guest submission / monitoring live in web-agent-seq-sft-submit.
---

# 分布式 train+eval 与断点续跑(2026-07-09 起)

支持两种模式,均基于 mini-web-agent 自带 OM2W harness(vllm serve + agent
循环,**非 SkyRL**):

- **A. train + 多节点 eval 一条 job**(训练可从 ckpt 续):训练用几个节点,
  训完这些节点就地并行 eval。
- **B. 独立多节点 eval**(断点可续):评任意 PVC 上的 HF ckpt。

## 运行协议(提交前用 AskUserQuestion 问清)

1. **NODES?**(续训时必须与原 run 一致,epoch/LR 折算假设 global batch 不变)
2. **是全新训练、续训、还是只 eval?** 续训要 checkpoint-N 路径 + 目标总
   epoch;只 eval 要 ckpt 路径(须已 vision merge)。
3. **eval 范围/并发?** `TASK_LEVEL`(all=300/easy80/medium143/hard77)、
   `TOTAL_WORKERS`(默认 80=browserbase 安全水位,加大前确认配额)。
4. **p0/p1 与 PRIORITY_CLASS_NAME**(同 web-agent-seq-sft-submit 的规则)。

## 三条命令

```bash
# A1. 全新训练 + 训完多节点并行全量 eval(一条 job)
NODES=4 RUN_NAME=<唯一训练名> SFT_CONFIG=examples/train_full/<yaml> \
WANDB_PROJECT=web-agent-sft AZBLOB_AUTO_PUSH=0 \
bash docker/submit_sft_eval_q35_image.sh

# A2. 续训版:同上,加两个 env(job 被杀后从 checkpoint-N 接着训)
RESUME_FROM_CKPT=/mnt/pvc/<alias>/code/mini-web-agent/LlamaFactory/saves/<output_dir>/checkpoint-<N> \
TARGET_TOTAL_EPOCHS=4 \
NODES=4 SFT_CONFIG=examples/train_full/<原yaml> ... bash docker/submit_sft_eval_q35_image.sh
# (纯训练不带 eval 用 submit_sft_q35_image.sh,同样支持这两个 env)

# B. 独立多节点 eval;断点续评 = 原样重提同一 EVAL_RUN_ID
EVAL_CKPT=/mnt/pvc/<alias>/models/<已merge的ckpt> NODES=4 TASK_LEVEL=all \
bash docker/submit_dist_eval_q35_image.sh
# 重跑失败任务(result.json 带 run_exception 的):前面加 RETRY_FAILED=1
```

**首次上全量前先冒烟**(几分钟,验证 RANK 注入/分片/barrier/judge 全链):

```bash
EVAL_CKPT=<ckpt> NODES=4 TASK_LEVEL=easy LIMIT=8 TOTAL_WORKERS=8 \
EVAL_RUN_ID=dist_eval_smoke1 bash docker/submit_dist_eval_q35_image.sh
# 日志核对:4 个 pod 各打出 shard=K/4;各 Selected 2 task(s);
# master 有 "shard rcs: 0 0 0 0" + judge 汇总;job 最终 Completed。
```

## 机制(一屏读懂)

- **分片**:任务文件顺序固定,每 pod 算 `idx % WORLD_SIZE == RANK`
  (volcano pytorch 插件注入 RANK/WORLD_SIZE),分片互不重叠、无通信。
- **推理**:每节点本地 `vllm serve`(tp=8,自动挂
  `configs/qwen3_5_train_aligned.jinja` + sliding window 48000)。
- **汇合**:所有分片写同一 PVC 目录;干完写 `shards/shard_K_of_N.done`
  (内容含 JOB_NAME+rc,旧 job 残留不误触发);master 等齐后 `--judge-only`
  统一判分。
- **续训(warm restart)**:ckpt 是 save_only_model,真 resume 不可能;
  `docker/prepare_warm_restart.py` 在 job 里自动 备份→vision merge→按
  trainer_state.json 折算(剩余 epoch = 目标 − 已训;LR 从中断值接 cosine,
  warmup 1%)→生成 `_cont<N>` yaml,全 rank 共用。幂等,续训 job 再被杀可原样重提。
- **续评**:完成判据 = `outputs/<task_id>/result.json`;重提同 EVAL_RUN_ID
  跳过已完成任务;om2w_judge 自身也增量(跳过已判 task_id)。
- **volcano 策略坑(已内置修复,别改回)**:job 策略是 TaskCompleted→
  CompleteJob、PodFailed→AbortJob,worker 先退/非零退会杀掉还在干活的
  master。所以 worker 一律等 master 完成标记(`job_complete.$JOB_NAME` /
  `.master_done`)后**恒以 0 退出**,分片失败靠 done 文件里的 rc 传给 master
  聚合进它的退出码。

## 结果路径

`RUN_ROOT = /mnt/pvc/<alias>/evals/<EVAL_RUN_ID>/`(模式 A 的 EVAL_RUN_ID
默认 = JOB_NAME):

| 内容 | 路径 |
|---|---|
| **最终总分 + level breakdown** | `RUN_ROOT/logs/<EVAL_RUN_ID>/run_summary_judge.json` |
| 逐任务判分明细 | `RUN_ROOT/outputs_eval_1/WebJudge_..._auto_eval_results.json` |
| 逐任务轨迹/产物 | `RUN_ROOT/outputs/<task_id>/`(result.json = 完成判据) |
| 各分片生成汇总 | `RUN_ROOT/logs/<EVAL_RUN_ID>/generation_summary_shardKofN.json` |
| 各节点 vLLM 日志 | `RUN_ROOT/logs/vllm_shardK.log` |

## 约束与坑

- **续训 NODES 必须与原 run 一致**;`TARGET_TOTAL_EPOCHS` 是含已训部分的总
  目标,不填=只补完原 yaml 计划的 epoch。
- 独立 eval 的 ckpt 必须已 vision merge(`vision.safetensors` 在);模式 A
  链式 eval 用的是 master 刚 sync+merge 的 ckpt,自动满足。
- 总 browserbase 并发 = TOTAL_WORKERS(自动均分到节点),80 是验证过的水位。
- sliding window 对 aria/URL 密集内容会低估 token,少数任务仍可能 vLLM 400
  → `RETRY_FAILED=1` 重提补跑。
- 某 pod 挂死不写 done:master 等 `EVAL_BARRIER_TIMEOUT`(4h)后照样 judge
  (缺失按 fail 计),重提同 RUN_ID 补跑。
- judge key/browserbase 凭据来自 webchain secret(提交脚本自动从
  `/data/t-yifeili/webchain_sampling/cred.sh` 创建挂载)。

## 关键文件

| 路径 | 作用 |
|---|---|
| `docker/submit_sft_eval_q35_image.sh` | 模式 A 提交(train→多节点 eval;EVAL_BACKEND=skyrl 回旧链) |
| `docker/submit_dist_eval_q35_image.sh` | 模式 B 提交(独立多节点 eval/续评) |
| `docker/run_dist_eval_q35_image.sh` | in-pod:每节点 vLLM+分片生成,master judge 汇总 |
| `docker/prepare_warm_restart.py` | in-pod:一键续训准备(备份+merge+折算生成 yaml) |
| `src/miniswewebagent/run/benchmarks/om2w.py` | harness 入口(--num-shards/--resume/--retry-failed/--judge-only) |
| `LlamaFactory/examples/train_full/*_cont*.yaml` | 手写续训 yaml 的参考例子 |

手工 warm-restart 折算(特殊情况用):备份 ckpt → vision merge → 改 yaml
(`model_name_or_path`=备份、`num_train_epochs`=目标−已训、`learning_rate`=
trainer_state.json 里最后的 LR、warmup 0.01、output_dir/run_name 加 `_cont<N>`)
→ 按原 NODES 提交。数据准备/PVC 上传/多用户/Guest 提交/监控见
web-agent-seq-sft-submit。
