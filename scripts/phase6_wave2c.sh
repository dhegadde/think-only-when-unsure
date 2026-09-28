#!/bin/bash
# Phase 6 wave 2 confirmation: (1) E2B at LR 5e-5 isolates the LR confound in run A; (2) E4B seeds 2 and 3.
# All: normalized head, Phase 5 mix, 1,500 steps; eval end-of-schedule (the pre-registered primary).
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/phase6; mkdir -p $OUT
run() { n=$1 m=$2 seed=$3; shift 3; d=$OUT/$n
  [ -f $d/result.json ] || { python -u -m typed_decisions.phase5_multitask train --seed $seed --head_norm --lr 5e-5 \
      --data_dir processed_data/phase5 --model $m --out $d "$@" > $d.log 2>&1 \
      && echo "train $n done $(date -u +%H:%M:%S)" || { echo "train $n FAILED"; exit 1; }; }
  [ -f $OUT/eval_$n.json ] || { python -u -m typed_decisions.phase5_multitask eval --head_norm --name $n --arm head \
      --model $m --adapter $d/adapter --head $d/head.pt --out $OUT/eval_$n.json > $OUT/eval_$n.log 2>&1 \
      && echo "eval $n done $(date -u +%H:%M:%S)" || { echo "eval $n FAILED"; exit 1; }; }; }
run e2b_lr5 google/gemma-4-E2B 1
run e4b_stab_s2 google/gemma-4-E4B 2 --grad_ckpt
run e4b_stab_s3 google/gemma-4-E4B 3 --grad_ckpt
echo "ALL DONE $(date -u +%H:%M:%S)"
