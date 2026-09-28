#!/usr/bin/env python3
"""Phase 7 analysis: accuracy / Brier / tokens per arm on the shared subsample, and the head -> reasoner cascade
(head answers when its top probability >= t, else the reasoner's answer is used). Runs on the Mac.

  python -m typed_decisions.phase7_analyze --dir runs/phase7
"""
import argparse
import json
from pathlib import Path

from typed_decisions.multitask_bench import HELDOUT_TASKS, load
from typed_decisions.phase7_generate import subsample


def brier(p, y):
    return sum((pi - (i == y)) ** 2 for i, pi in enumerate(p))


def head_tokens(it, tok):
    """Tokens the head reads today (K full rows) and with a shared prefix (prefix once + options)."""
    prefix = len(tok(f"{it.state}\n\nQuestion: {it.instruction}\nAnswer:")["input_ids"])
    opts = [len(tok(" " + o, add_special_tokens=False)["input_ids"]) + 1 for o in it.options]
    return sum(prefix + o for o in opts), prefix + sum(opts)


def load_arm(path):
    return {(r["task"], r["idx"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/phase7")
    ap.add_argument("--data_dir", default="processed_data/phase5")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    ap.add_argument("--reasoners", nargs="+", default=["direct_e4b_it", "think_e4b_it"])
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    d = Path(args.dir)
    head_p = json.loads((d / "head_e4b_probs.json").read_text())
    arms = {n: load_arm(d / f"{n}.jsonl") for n in args.reasoners if (d / f"{n}.jsonl").exists()}
    rows = []  # one per subsampled item
    for t in HELDOUT_TASKS:
        items = load(f"test_{t}", args.data_dir)
        for i in subsample(items):
            it = items[i]
            full, shared = head_tokens(it, tok)
            r = {"task": t, "label": it.label, "head": head_p[t][i], "head_tok": full, "head_tok_shared": shared}
            for n, a in arms.items():
                if (t, i) in a:
                    r[n] = a[(t, i)]
            rows.append(r)
    out = {"n_items": len(rows), "arms": {}, "per_task": {}, "cascade": {}}

    def summarize(sel, get_p, get_tok):
        # task-balanced average (each task weighs the same, as in earlier phases)
        accs, brs, toks = [], [], []
        for t in HELDOUT_TASKS:
            rs = [r for r in sel if r["task"] == t]
            if not rs:
                continue
            accs.append(sum(max(range(len(get_p(r))), key=get_p(r).__getitem__) == r["label"] for r in rs) / len(rs))
            brs.append(sum(brier(get_p(r), r["label"]) for r in rs) / len(rs))
            toks.append(sum(get_tok(r) for r in rs) / len(rs))
        return {"acc": sum(accs) / len(accs), "brier": sum(brs) / len(brs), "tokens": sum(toks) / len(toks)}

    out["arms"]["head"] = summarize(rows, lambda r: r["head"], lambda r: r["head_tok"])
    out["arms"]["head"]["tokens_shared_prefix"] = summarize(rows, lambda r: r["head"], lambda r: r["head_tok_shared"])["tokens"]
    for n in arms:
        sel = [r for r in rows if n in r]
        out["arms"][n] = summarize(sel, lambda r: r[n]["probs"], lambda r: r[n]["tokens_in"] + r[n]["tokens_out"])
        out["arms"][n]["n"] = len(sel)
        out["arms"][n]["forced_rate"] = sum(r[n]["forced"] for r in sel) / max(1, len(sel))
        out["arms"][n]["tokens_out"] = sum(r[n]["tokens_out"] for r in sel) / max(1, len(sel))
    for t in HELDOUT_TASKS:
        rs = [r for r in rows if r["task"] == t]
        out["per_task"][t] = {n: sum(max(range(len(p)), key=p.__getitem__) == r["label"]
                                     for r in rs for p in [r["head"] if n == "head" else r[n]["probs"]]
                                     ) / len(rs) for n in ["head", *[a for a in arms if all(a in r for r in rs)]]}
    # Cascade: head first (tokens = shared-prefix head cost), escalate below threshold t to each reasoner.
    for n in arms:
        sel = [r for r in rows if n in r]
        curve = []
        for thr in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]:
            esc = lambda r: max(r["head"]) < thr
            s = summarize(sel, lambda r: r[n]["probs"] if esc(r) else r["head"],
                          lambda r: r["head_tok_shared"] + (r[n]["tokens_in"] + r[n]["tokens_out"] if esc(r) else 0))
            s["threshold"], s["escalated"] = thr, sum(map(esc, sel)) / len(sel)
            curve.append(s)
        out["cascade"][n] = curve
    (d / "analysis.json").write_text(json.dumps(out, indent=2))
    print(f"{out['n_items']} items")
    for n, s in out["arms"].items():
        print(f"{n:16} acc {s['acc']*100:5.1f}  brier {s['brier']:.3f}  tokens/item {s['tokens']:7.0f}"
              + (f"  (shared-prefix {s['tokens_shared_prefix']:.0f})" if n == "head" else
                 f"  out {s['tokens_out']:.0f}  forced {s['forced_rate']*100:.1f}%"))
    print("per task:", json.dumps({t: {k: round(v * 100, 1) for k, v in d_.items()} for t, d_ in out["per_task"].items()}))
    for n, curve in out["cascade"].items():
        print(f"cascade head -> {n}:")
        for s in curve:
            print(f"  t={s['threshold']:.2f} escalate {s['escalated']*100:5.1f}%  acc {s['acc']*100:5.1f}  tokens {s['tokens']:7.0f}")


if __name__ == "__main__":
    main()
