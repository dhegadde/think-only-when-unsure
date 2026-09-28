#!/usr/bin/env python3
"""Phase 7: the same held-out decision questions answered by an instruction-tuned model, directly or after its
native thinking, with token counts. Plan: goals/typed-decision-heads/plans/phase7_options.md (Option 1).

  direct  : user turn = question + lettered options + "Reply with only the letter"; no thinking.
  think   : same user turn, Gemma 4 thinking mode on (<|think|> system turn); thoughts end at <channel|>.
The answer distribution is read at the first answer position, restricted to the option letters (both the
bare and the space-prefixed letter token), so every arm yields option probabilities for Brier and cascades.

  python -m typed_decisions.phase7_generate --mode direct --out runs/phase7/direct_e4b.jsonl
"""
import argparse
import json
import random
import time
from pathlib import Path

from typed_decisions.multitask_bench import HELDOUT_TASKS, load

LETTERS = "ABCDEFGHIJ"
PER_TASK = 200  # fixed subsample of each held-out task (all arms scored on the same items)
DEV_N = 400  # fixed subsample of the (in-distribution) dev set


def subsample(items, n=PER_TASK, seed=0):
    """Indices (into the test file) of a fixed random subsample, sorted."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    return sorted(idx[:n])


def user_turn(it):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it.options))
    return (f"{it.state}\n\nQuestion: {it.instruction}\nOptions:\n{opts}\n\n"
            "Reply with only the letter of the correct option.")


def letter_ids(tok, k):
    """For each of the first k letters: token ids of 'X' and ' X' (single tokens in Gemma's vocabulary)."""
    out = []
    for L in LETTERS[:k]:
        ids = {tuple(tok(L, add_special_tokens=False)["input_ids"]), tuple(tok(" " + L, add_special_tokens=False)["input_ids"])}
        out.append([i[0] for i in ids if len(i) == 1])
    return out


def answer_probs(logits, lids):
    """Option probabilities from one position's logits: logsumexp over each letter's token variants, softmax."""
    import torch
    lp = torch.log_softmax(logits.float(), dim=-1)
    s = torch.stack([torch.logsumexp(lp[ids], 0) for ids in lids])
    return torch.softmax(s, 0).tolist()


def main():
    import torch
    from typed_decisions.gemma_core import load_gemma

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("direct", "think"), required=True)
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_think", type=int, default=1024, help="thinking tokens before the answer is forced")
    p.add_argument("--data_dir", default="processed_data/phase5")
    p.add_argument("--split", choices=("test", "dev"), default="test")
    p.add_argument("--per_task", type=int, default=PER_TASK, help="test items per task (a prefix-subset of the default)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if torch.cuda.device_count() > 1:  # a model too big for one GPU (Gemma 4 31B): layers spread over all GPUs
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    else:
        model, tok = load_gemma(torch.device("cuda"), model_name=args.model)  # text-only, towers dropped
    model.eval()
    tok.padding_side = "left"
    close = tok("<channel|>", add_special_tokens=False)["input_ids"]
    all_letter_ids = {i for ids in letter_ids(tok, len(LETTERS)) for i in ids}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task"], r["idx"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = []
    if args.split == "dev":  # in-distribution dev items (training tasks), for choosing the cascade threshold
        items = load("dev", args.data_dir)
        todo = [(items[i].task, i, items[i]) for i in subsample(items, n=DEV_N) if (items[i].task, i) not in done]
    for t in HELDOUT_TASKS if args.split == "test" else ():
        items = load(f"test_{t}", args.data_dir)
        todo += [(t, i, items[i]) for i in subsample(items, n=args.per_task) if (t, i) not in done]
    todo.sort(key=lambda x: -len(x[2].state))  # similar lengths per batch; OOM shows up early
    print(f"{args.mode}: {len(todo)} items to do ({len(done)} done)", flush=True)
    t0 = time.time()
    with out.open("a") as f:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b : b + args.batch]
            prompts = [tok.apply_chat_template([{"role": "user", "content": user_turn(it)}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=args.mode == "think")
                       for _, _, it in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            n_in = enc["attention_mask"].sum(1).tolist()
            budget = args.max_think + 16 if args.mode == "think" else 16
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=budget, do_sample=False)
            gen = g[:, enc["input_ids"].shape[1] :].tolist()
            for (t, i, it), p_ids, ni, gi in zip(chunk, prompts, n_in, gen):
                gi = [x for x in gi if x != tok.pad_token_id]
                # Answer position: first letter token after the thinking block (or anywhere, in direct mode).
                start = 0
                if args.mode == "think":
                    for j in range(len(gi) - len(close) + 1):
                        if gi[j : j + len(close)] == close:
                            start = j + len(close)
                            break
                    else:
                        start = None  # thinking did not finish within the budget
                pos = next((j for j in range(start, len(gi)) if gi[j] in all_letter_ids), None) if start is not None else None
                forced = pos is None
                if forced:  # score the letter right after an explicit end of thinking / answer prefix
                    prefix = gi[: start if start else len(gi)] + (close if args.mode == "think" and start is None else [])
                else:
                    prefix = gi[:pos]
                ids = torch.tensor([tok(p_ids, add_special_tokens=False)["input_ids"] + prefix], device="cuda")
                with torch.no_grad():
                    logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
                probs = answer_probs(logits, letter_ids(tok, len(it.options)))
                think_text = tok.decode(gi[: start - len(close)] if start else gi, skip_special_tokens=True) \
                    if args.mode == "think" else ""
                f.write(json.dumps({"task": t, "idx": i, "label": it.label, "probs": probs, "tokens_in": ni,
                                    "tokens_out": len(prefix) + 1, "forced": forced,
                                    "think_chars": len(think_text)}) + "\n")
            f.flush()
            print(f"{min(b + args.batch, len(todo))}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
