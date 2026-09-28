# Think Only When Unsure

Code, data and per-question results for **"Think Only When Unsure: One-Pass Confidence Routing Matches Reasoning
at Half the Tokens on Decision Tasks"** (Dheeraj Gadde, 2026). Paper: [`paper/paper.pdf`](paper/paper.pdf).

On 10 decision tasks the models never trained on (1,900 questions), answering in one pass when confident and
thinking only when unsure matched always-thinking accuracy at about half the tokens:

| Approach (Gemma 4) | Accuracy | Tokens / question |
|---|---|---|
| E4B + trained decision head (one pass) | 77.2% | 93 |
| E4B-it, one-letter answer | 77.4% | 121 |
| E4B-it, thinking first | 80.8% | 521 |
| **Head first, E4B-it thinking only when unsure (3 seeds, threshold chosen on dev)** | **80.7 ± 1.1%** | **259** |
| 31B-it, one-letter answer | 85.7% | 124 |
| 31B-it one letter, thinking only on its least-sure 10% (600 items) | 85.0% | 207 |
| 31B-it, thinking first (600 items) | 85.8% | 495 |

Every prediction was written down before its run; the ones that turned out wrong are listed in the paper's
appendix.

## Layout

| Path | What |
|---|---|
| `typed_decisions/decision_head.py` | the one-pass decision head (MLP over the final hidden state) |
| `typed_decisions/gemma_core.py` | text-only Gemma 4 loading, hidden states, readout |
| `typed_decisions/multitask_bench.py` | decision items: 6 training tasks, 10 held-out tasks |
| `typed_decisions/phase5_multitask.py` | `train` (LoRA + head) and `eval` (held-out tasks, per-item probabilities, shared-prefix scoring) |
| `typed_decisions/phase7_generate.py` | instruction-tuned model answering directly or with native thinking, with token counts |
| `typed_decisions/phase7_analyze.py` | accuracy, Brier, tokens and cascade curves from the saved outputs |
| `typed_decisions/phase5_jev.py` | the Jev (TypeSafe AI) API comparison (needs your own API key) |
| `scripts/` | the exact GPU run scripts, in order: `phase6_wave2a.sh` (E4B head, seed 1), `phase6_wave2c.sh` (E2B control, seeds 2-3), `phase7_a.sh` ... `phase7_d.sh` |
| `processed_data/phase5/` | the decision items used (see `NOTICE.md` for sources) |
| `results/` | per-question outputs and summaries for every number in the paper |
| `paper/` | LaTeX source, bibliography, figure script |

## Reproduce

Analysis from the saved results (no GPU):

```bash
pip install -r requirements.txt
python -m typed_decisions.phase7_analyze --dir results/phase7
python -m unittest discover -s tests
```

Full runs need a 32 GB GPU for the E4B head (about 35-70 minutes per seed) and about 96 GB spread over several
GPUs for Gemma 4 31B. Run the scripts in `scripts/` from the repository root; they write to `runs/`. The whole
study cost about $10 of rented GPU time.

## License

MIT (see `LICENSE`); third-party code and data are listed in `NOTICE.md`.
