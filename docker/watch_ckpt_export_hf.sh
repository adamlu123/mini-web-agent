#!/usr/bin/env bash
# Watcher: convert every new SkyRL ckpt (global_step_N) to HF safetensors.
#
# Runs as a CPU-only volcano job (or anywhere with the PVC mounted + torch).
# Loops: read latest_ckpt_global_step.txt -> if that step has a complete,
# quiescent policy dir and no HF export yet -> run merge_skyrl_ckpt_to_hf.py.
# ckpts roll fast (max_ckpts_to_keep=2, ~44 min/step), so the poll interval
# must stay well under one step; missed/pruned steps are skipped gracefully.
#
# Env:
#   OUT_BASE   e.g. /mnt/pvc/t-yifeili/outputs/t-yifeili-p1-cua-job-c1676
#   BASE_HF    HF dir for config/tokenizer (the SFT init ckpt)
#   MERGER     path to merge_skyrl_ckpt_to_hf.py
#   INTERVAL   poll seconds (default 120)
#   KEEP_LAST  keep only the newest K HF exports, 0 = keep all (default 0)
#   QUIESCE_S  file quiescence threshold before merging (default 90)
set -u

OUT_BASE="${OUT_BASE:?OUT_BASE not set}"
BASE_HF="${BASE_HF:?BASE_HF not set}"
MERGER="${MERGER:?MERGER not set}"
INTERVAL="${INTERVAL:-120}"
KEEP_LAST="${KEEP_LAST:-0}"
QUIESCE_S="${QUIESCE_S:-90}"

CKPTS="$OUT_BASE/ckpts"
HF_ROOT="$OUT_BASE/exports/hf"
mkdir -p "$HF_ROOT"
echo "[watch] OUT_BASE=$OUT_BASE KEEP_LAST=$KEEP_LAST INTERVAL=${INTERVAL}s"

complete_and_quiet() {  # $1 = policy dir
  local pdir="$1"
  [ -d "$pdir" ] || return 1
  local world file_count newest_age
  world=$(ls "$pdir" 2>/dev/null | grep -oE 'world_size_[0-9]+' | head -1 | grep -oE '[0-9]+')
  [ -n "$world" ] || return 1
  file_count=$(ls "$pdir" 2>/dev/null | grep -cE "^model_world_size_${world}_rank_[0-9]+\.pt$")
  [ "$file_count" -eq "$world" ] || return 1
  newest_age=$(( $(date +%s) - $(stat -c %Y "$pdir"/model_*.pt 2>/dev/null | sort -n | tail -1) ))
  [ "$newest_age" -ge "$QUIESCE_S" ]
}

while true; do
  step=$(cat "$CKPTS/latest_ckpt_global_step.txt" 2>/dev/null || true)
  if [ -n "$step" ] && [ ! -d "$HF_ROOT/global_step_$step" ]; then
    pdir="$CKPTS/global_step_$step/policy"
    if complete_and_quiet "$pdir"; then
      echo "[watch] exporting global_step_$step ..."
      if nice -n 15 python "$MERGER" \
          --policy-dir "$pdir" \
          --base "$BASE_HF" \
          --out "$HF_ROOT/global_step_$step"; then
        echo "[watch] global_step_$step -> HF done"
      else
        echo "[watch] WARN: merge failed for global_step_$step (rc=$?); will retry next loop"
        rm -rf "$HF_ROOT/global_step_$step.tmp"
      fi
      if [ "$KEEP_LAST" -gt 0 ]; then
        ls -d "$HF_ROOT"/global_step_* 2>/dev/null | sort -t_ -k3 -n | head -n -"$KEEP_LAST" | while read -r old; do
          echo "[watch] pruning $old (KEEP_LAST=$KEEP_LAST)"
          rm -rf "$old"
        done
      fi
    fi
  fi
  sleep "$INTERVAL"
done
