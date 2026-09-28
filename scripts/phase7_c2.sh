#!/bin/bash
# Phase 7 run C, trimmed for cost (31B thinking ~3 min per 16 items): thinking on 60 items per task (a subset of
# the 200-per-task sample; subsample() prefixes nest), no dev run. Appends to the same driver log.
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/phase7
python -u -m typed_decisions.phase7_generate --model google/gemma-4-31B-it --mode think --split test --per_task 60 \
  --out $OUT/think_31b_it.jsonl --batch 16 >> $OUT/think_31b_it.log 2>&1 \
  && echo "think_31b_it (60/task) done $(date -u +%H:%M:%S)" || { echo "think_31b_it FAILED"; exit 1; }
echo "ALL DONE $(date -u +%H:%M:%S)"
