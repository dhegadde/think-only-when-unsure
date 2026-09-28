# Third-party code and data

- `training/schedules.py` is taken unchanged from [llm-research-kit](https://github.com/vukrosic/llm-research-kit)
  (MIT License, Copyright (c) 2025 Vuk Rosić). The rest of this repository was written for this study, inside a
  working copy of that kit.
- `processed_data/phase5/` holds decision items converted from public datasets (BoolQ, SST-2, MNLI,
  CommonsenseQA, Yelp polarity, IMDB, RTE, MRPC, STS-B, HellaSwag, ARC-Easy, OpenBookQA, WiC, Winogrande, COPA),
  loaded through Hugging Face `datasets`; each remains under its original license. The `coherence` items were
  generated from the kit's pretraining text shard; the builder for them (`coherence_items` in
  `typed_decisions/multitask_bench.py`) needs the kit and is kept for reference only.
- Model weights (Google Gemma 4, Apache 2.0) are downloaded from Hugging Face and are not included.
- Jev (TypeSafe AI) results in `results/phase5/` are aggregate scores only; raw API responses are not included.
