#!/bin/bash
# Phase 7 run A: (1) per-item head probabilities of the E4B-stab head (seed 1) on the 10 held-out tasks;
# (2) Gemma 4 E4B-it answering directly; (3) E4B-it with native thinking. Same items (200/task subsample) for 2-3.
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/phase7; mkdir -p $OUT
[ -f $OUT/head_e4b_probs.json ] || { python -u -m typed_decisions.phase5_multitask eval --head_norm --name head_e4b --arm head \
    --model google/gemma-4-E4B --adapter runs/phase6/e4b_stab/adapter --head runs/phase6/e4b_stab/head.pt \
    --out $OUT/eval_head_e4b.json --save_probs $OUT/head_e4b_probs.json > $OUT/head_e4b.log 2>&1 \
    && echo "head probs done $(date -u +%H:%M:%S)" || { echo "head probs FAILED"; exit 1; }; }
for mode in direct think; do
  python -u -m typed_decisions.phase7_generate --mode $mode --out $OUT/${mode}_e4b_it.jsonl > $OUT/${mode}_e4b_it.log 2>&1 \
    && echo "$mode done $(date -u +%H:%M:%S)" || { echo "$mode FAILED"; exit 1; }
done
echo "ALL DONE $(date -u +%H:%M:%S)"
