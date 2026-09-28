#!/bin/bash
# Phase 7 run B: (1) head probs for E4B-stab seeds 2, 3 on the held-out tests; (2) head probs for seeds 1-3 on the
# in-distribution dev set; (3) E4B-it thinking on a 400-item dev subsample (threshold selection on dev, not test);
# (4) shared-prefix head (seed 1) on the held-out tests: same probabilities? how much faster?
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/phase7; mkdir -p $OUT
ev() { n=$1; shift; [ -f $OUT/$n.done ] && return 0
  python -u -m typed_decisions.phase5_multitask eval --head_norm --arm head --model google/gemma-4-E4B --name $n "$@" \
    > $OUT/$n.log 2>&1 && touch $OUT/$n.done && echo "$n done $(date -u +%H:%M:%S)" || { echo "$n FAILED"; exit 1; }; }
for s in 2 3; do
  ev head_e4b_s$s --adapter runs/phase6/e4b_stab_s$s/adapter --head runs/phase6/e4b_stab_s$s/head.pt \
    --out $OUT/eval_head_e4b_s$s.json --save_probs $OUT/head_e4b_s${s}_probs.json
done
for s in 1 2 3; do
  d=runs/phase6/e4b_stab; [ $s = 1 ] || d=runs/phase6/e4b_stab_s$s
  ev dev_head_e4b_s$s --split dev --adapter $d/adapter --head $d/head.pt --out $OUT/eval_dev_s$s.json \
    --save_probs $OUT/dev_head_e4b_s${s}_probs.json
done
python -u -m typed_decisions.phase7_generate --mode think --split dev --out $OUT/think_e4b_it_dev.jsonl \
  > $OUT/think_e4b_it_dev.log 2>&1 && echo "think dev done $(date -u +%H:%M:%S)" || { echo "think dev FAILED"; exit 1; }
ev head_e4b_shared --shared_prefix --adapter runs/phase6/e4b_stab/adapter --head runs/phase6/e4b_stab/head.pt \
  --out $OUT/eval_head_e4b_shared.json --save_probs $OUT/head_e4b_shared_probs.json
echo "ALL DONE $(date -u +%H:%M:%S)"
