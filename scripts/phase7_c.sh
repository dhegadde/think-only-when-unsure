#!/bin/bash
# Phase 7 run C: bigger reasoner, Gemma 4 31B-it (layers over all visible GPUs), direct and thinking, on the same
# 1,900 held-out items and the 400 dev items (for the dev-chosen cascade threshold).
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/phase7; mkdir -p $OUT
M=google/gemma-4-31B-it
gen() { python -u -m typed_decisions.phase7_generate --model $M --mode $1 --split $2 --out $OUT/$3.jsonl --batch 16 \
    > $OUT/$3.log 2>&1 && echo "$3 done $(date -u +%H:%M:%S)" || { echo "$3 FAILED"; exit 1; }; }
gen direct test direct_31b_it
gen think test think_31b_it
gen think dev think_31b_it_dev
echo "ALL DONE $(date -u +%H:%M:%S)"
