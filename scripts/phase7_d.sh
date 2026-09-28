#!/bin/bash
# Phase 7 run D: batched shared-prefix head scoring (seed 1) on all held-out tests; compare to K-row probs + time.
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/phase7
python -u -m typed_decisions.phase5_multitask eval --head_norm --arm head --model google/gemma-4-E4B --name head_e4b_sharedb \
  --shared_prefix batched --adapter runs/phase6/e4b_stab/adapter --head runs/phase6/e4b_stab/head.pt \
  --out $OUT/eval_head_e4b_sharedb.json --save_probs $OUT/head_e4b_sharedb_probs.json > $OUT/head_e4b_sharedb.log 2>&1 \
  && echo "sharedb done $(date -u +%H:%M:%S)" || echo "sharedb FAILED"
python -u -m typed_decisions.phase5_multitask eval --head_norm --arm head --model google/gemma-4-E4B --name head_e4b_krows_t \
  --adapter runs/phase6/e4b_stab/adapter --head runs/phase6/e4b_stab/head.pt \
  --out $OUT/eval_head_e4b_krows_t.json > $OUT/head_e4b_krows_t.log 2>&1 && echo "krows timing done $(date -u +%H:%M:%S)"
echo "ALL DONE $(date -u +%H:%M:%S)"
