#!/usr/bin/env python3
"""Phase 5 reference arm: Jev (TypeSafe API) zero-shot on the held-out decision tasks, with exactly the same
state / instruction / options text as the Gemma arms. Every task is sent as a Choice question (yes/no as
the two options "Yes"/"No", score tasks as their described levels), so all arms answer the same form.
Resumable: results are appended per item to a JSONL cache. Key read from a file, never printed.

Usage: python -m typed_decisions.phase5_jev --key-file <path> --out runs/phase5/jev.json [--limit N]
"""
import argparse
import json
import time
from pathlib import Path

import requests

API_URL = "https://api.typesafe.ai/v1/systemone"  # TypeSafe AI Jev endpoint; the API key is read from --key-file
from typed_decisions.multitask_bench import HELDOUT_TASKS, load
from typed_decisions.phase5_multitask import metrics, NEAR, MEDIUM, FAR

PRICE_PER_M_INPUT = 0.042  # USD, confirmed 2026-09-22 (typed_decisions/jev_compare.py)


def ask(key, it, max_retries=5):
    keys = [f"option_{i}" for i in range(len(it.options))]
    payload = {"state": it.state, "model": "jev-latest",
               "questions": {"q": {"type": "choice", "instructions": it.instruction,
                                   "criteria": dict(zip(keys, it.options))}}}
    backoff = 1.0
    for _ in range(max_retries):
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json=payload, timeout=60)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        d = r.json()
        return [d["answers"]["q"]["probabilities"][k] for k in keys], d["usage"]["input_tokens"]
    raise RuntimeError("rate-limited")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    key = args.key_file.read_text().strip()
    cache_path = args.out.with_suffix(".cache.jsonl")
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            c = json.loads(line)
            cache[(c["task"], c["i"])] = c
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res, tokens = {}, sum(c["tokens"] for c in cache.values())
    with open(cache_path, "a") as f:
        for t in HELDOUT_TASKS:
            items = load(f"test_{t}")[: args.limit]
            probs = []
            for i, it in enumerate(items):
                if (t, i) not in cache:
                    p, n = ask(key, it)
                    cache[(t, i)] = {"task": t, "i": i, "probs": p, "tokens": n}
                    f.write(json.dumps(cache[(t, i)]) + "\n")
                    f.flush()
                    tokens += n
                probs.append(cache[(t, i)]["probs"])
            res[t] = metrics(probs, items)
            print(f"jev {t}: acc {res[t]['acc']:.3f} brier {res[t]['brier']:.3f}  "
                  f"(running cost ${tokens / 1e6 * PRICE_PER_M_INPUT:.4f})", flush=True)
    for g, ts in (("near", NEAR), ("medium", MEDIUM), ("far", FAR), ("all", set(HELDOUT_TASKS))):
        res[f"avg_{g}"] = {k: sum(res[t][k] for t in ts) / len(ts) for k in ("acc", "brier")}
    args.out.write_text(json.dumps({"arm": "jev", "input_tokens": tokens,
                                    "est_cost_usd": round(tokens / 1e6 * PRICE_PER_M_INPUT, 4), "results": res}, indent=2))


if __name__ == "__main__":
    main()
