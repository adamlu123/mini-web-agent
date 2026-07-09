#!/usr/bin/env python
"""一键 warm-restart 准备(在 pod 内由 run_sft_q35_image.sh 的 master 调用)。

训练 yaml 都是 save_only_model: true,checkpoint-* 里没有 optimizer/DeepSpeed
状态,resume_from_checkpoint 真断点续跑不可能。本脚本把"手工三步"自动化成
可重入的一步:

  1. 备份:把易失 saves/ 里的 checkpoint-N 拷到 --backup-root 下的稳定路径
     (先拷 .tmp 再原子 rename;备份已存在且完整则跳过 -> 重提同一 job 幂等);
  2. vision merge:文本 SFT ckpt 缺 vision tower,当 model_name_or_path 加载
     会随机初始化视觉塔;备份里没有 vision.safetensors 时自动从 HF 缓存的
     base 快照补全(scripts/merge_vision_from_base.py);
  3. 生成续训 yaml:以原训练 yaml 为模板 patch 掉五个字段——
        model_name_or_path -> 备份路径
        num_train_epochs   -> 目标总 epoch - ckpt 已训 epoch(trainer_state.json)
        learning_rate      -> ckpt 中断处的 LR(log_history 最后一条),从原
                              cosine 断点接着衰减,不做二次大 warmup
        warmup_ratio       -> 0.01
        output_dir/run_name -> 加 _cont<step> 后缀,不覆盖原 run
     写到 --out-config,driver 随后用它训练。

注意:剩余 epoch/LR 的折算隐含"续训的 global batch 与原 run 一致",即
NODES×GPUS×per_device×grad_accum 不变;NODES 变了折算就错(脚本无法自查,
调用方需保证)。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def log(msg: str) -> None:
    print(f"[warm-restart] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[warm-restart][error] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def has_weights(d: Path) -> bool:
    return (d / "model.safetensors").exists() or (d / "model.safetensors.index.json").exists()


def read_ckpt_state(ckpt: Path) -> tuple[float, float, int]:
    """返回 (已训 epoch, 中断处 LR, global_step)。"""
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        die(f"{state_path} 不存在;不是 trainer 存的 checkpoint-* 目录?")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    epoch_done = float(state.get("epoch") or 0.0)
    global_step = int(state.get("global_step") or 0)
    lr_last = 0.0
    for row in reversed(state.get("log_history", [])):
        if "learning_rate" in row:
            lr_last = float(row["learning_rate"])
            break
    if epoch_done <= 0 or global_step <= 0:
        die(f"trainer_state.json 里 epoch/global_step 异常: epoch={epoch_done} step={global_step}")
    if lr_last <= 0:
        die("log_history 里找不到 learning_rate;请手写续训 yaml 指定 LR")
    return epoch_done, lr_last, global_step


def ensure_backup(ckpt: Path, backup_root: Path, run_name: str, global_step: int) -> Path:
    backup = backup_root / f"{run_name}_ckpt{global_step}_bak"
    if str(ckpt.resolve()).startswith(str(backup_root.resolve())):
        log(f"ckpt 已在 backup-root 下,直接使用: {ckpt}")
        return ckpt
    if has_weights(backup):
        log(f"备份已存在,跳过拷贝: {backup}")
        return backup
    tmp = backup.with_name(backup.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    log(f"备份 ckpt(可能要几分钟): {ckpt} -> {backup}")
    backup_root.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True)
    # 不用 shutil.copytree/cp -a:跨属主/跨文件系统保留元数据会报
    # "Operation not supported";只拷内容
    proc = subprocess.run(
        ["cp", "-r", "--no-preserve=mode,ownership", f"{ckpt}/.", str(tmp)], text=True
    )
    if proc.returncode != 0 or not has_weights(tmp):
        die(f"备份拷贝失败: {ckpt} -> {tmp}")
    tmp.replace(backup)
    return backup


def ensure_vision(backup: Path, base_model_id: str, hf_home: Path, merge_script: Path) -> None:
    if (backup / "vision.safetensors").exists():
        log("vision.safetensors 已在,跳过 merge")
        return
    snap_root = hf_home / "hub" / f"models--{base_model_id.replace('/', '--')}" / "snapshots"
    snaps = sorted(snap_root.glob("*/")) if snap_root.exists() else []
    if not snaps:
        die(f"HF 缓存里找不到 base 快照: {snap_root}(先跑过一次同 base 的训练,或手动 merge)")
    base = snaps[0]
    log(f"vision merge: base={base}")
    proc = subprocess.run(
        [sys.executable, str(merge_script), "--ckpt", str(backup), "--base", str(base)],
        text=True,
    )
    if proc.returncode != 0 or not (backup / "vision.safetensors").exists():
        die("vision merge 失败;续训会随机初始化视觉塔,拒绝继续")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint-N 目录(saves/ 里的或已备份的)")
    ap.add_argument("--config", required=True, help="原训练 yaml(模板)")
    ap.add_argument("--out-config", required=True, help="生成的续训 yaml 路径")
    ap.add_argument("--backup-root", required=True, help="稳定备份根目录(如 /mnt/pvc/<alias>/models)")
    ap.add_argument("--merge-script", required=True, help="scripts/merge_vision_from_base.py 路径")
    ap.add_argument("--hf-home", required=True, help="HF 缓存(找 base 快照做 vision merge)")
    ap.add_argument("--target-total-epochs", type=float, default=0.0,
                    help="目标总 epoch(含已训部分);0 = 用原 yaml 的 num_train_epochs")
    ap.add_argument("--base-model-id", default="", help="vision merge 的 base(默认取原 yaml 的 model_name_or_path)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_dir() or not has_weights(ckpt):
        die(f"--ckpt 不是 HF 权重目录: {ckpt}")
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    epoch_done, lr_last, global_step = read_ckpt_state(ckpt)
    target_total = args.target_total_epochs or float(cfg.get("num_train_epochs") or 0)
    remaining = round(target_total - epoch_done, 2)
    if remaining <= 0:
        die(f"目标总 epoch {target_total} <= ckpt 已训 {epoch_done:.2f};用 TARGET_TOTAL_EPOCHS 提高目标")

    run_name = str(cfg.get("run_name") or Path(str(cfg.get("output_dir", "run"))).name)
    backup = ensure_backup(ckpt, Path(args.backup_root), run_name, global_step)

    base_model_id = args.base_model_id or str(cfg.get("model_name_or_path") or "")
    if not base_model_id or Path(base_model_id).is_absolute():
        # 原 yaml 已经指向本地目录(比如上一轮 cont),vision 权重来源仍是原始 base
        base_model_id = "Qwen/Qwen3.5-9B"
    ensure_vision(backup, base_model_id, Path(args.hf_home), Path(args.merge_script))

    cfg["model_name_or_path"] = str(backup)
    cfg["num_train_epochs"] = remaining
    cfg["learning_rate"] = lr_last
    cfg["warmup_ratio"] = 0.01
    cfg["output_dir"] = f"{cfg['output_dir']}_cont{global_step}"
    cfg["run_name"] = f"{run_name}_cont{global_step}"
    cfg["resume_from_checkpoint"] = None

    out = Path(args.out_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"### AUTO-GENERATED warm restart (prepare_warm_restart.py)\n"
        f"### from ckpt: {ckpt}\n"
        f"### epoch_done={epoch_done:.3f} target_total={target_total} -> remaining={remaining}\n"
        f"### lr resumes at {lr_last:.3e} (cosine to 0, warmup 1%)\n"
    )
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log(f"已训 {epoch_done:.2f} ep @ step {global_step} | 目标 {target_total} ep -> 续训 {remaining} ep | LR {lr_last:.3e}")
    log(f"续训 yaml: {out}")
    log(f"output_dir: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
