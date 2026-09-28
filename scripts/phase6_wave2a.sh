#!/bin/bash
# Phase 6 wave 2 run A: readout-scale diagnostic, then E4B stabilized (normalized head, grad ckpt, LR 5e-5,
# non-finite skipping, best-dev checkpoint). Eval end-of-schedule AND best-dev on the 10 held-out tasks.
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/phase6; mkdir -p $OUT
[ -s $OUT/readout_scale2.log ] && grep -q "E4B" $OUT/readout_scale2.log || \
  { python -u runpod_run/readout_scale.py > $OUT/readout_scale2.log 2>&1 && echo "diagnostic done $(date -u +%H:%M:%S)" || echo "diagnostic FAILED (continuing)"; }
d=$OUT/e4b_stab
[ -f $d/result.json ] || { python -u -m typed_decisions.phase5_multitask train --seed 1 --head_norm --lr 5e-5 --grad_ckpt \
    --data_dir processed_data/phase5 --model google/gemma-4-E4B --out $d > $d.log 2>&1 \
    && echo "train e4b_stab done $(date -u +%H:%M:%S)" || { echo "train e4b_stab FAILED"; exit 1; }; }
ev() { [ -f $OUT/eval_$1.json ] && return 0
  python -u -m typed_decisions.phase5_multitask eval --head_norm --name $1 --arm head --model google/gemma-4-E4B \
    --adapter $2/adapter --head $2/head.pt --out $OUT/eval_$1.json > $OUT/eval_$1.log 2>&1 \
    && echo "eval $1 done $(date -u +%H:%M:%S)" || { echo "eval $1 FAILED"; exit 1; }; }
ev e4b_stab $d
[ -d $d/best/adapter ] && ev e4b_stab_best $d/best
echo "ALL DONE $(date -u +%H:%M:%S)"
