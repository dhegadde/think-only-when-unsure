#!/usr/bin/env python3
"""Phase 6 wave 2: teacher rationales for rationale distillation (Hsieh et al. 2023, "Distilling step-by-step").

A teacher model sees a training item AND its correct answer and writes 1-3 sentences on why that answer is
correct (label-conditioned, so a rationale never argues for a wrong answer). The student is trained to write
it as an extra next-token loss; at test time only the decision head is used (one pass, no generated tokens).
Plan: goals/typed-decision-heads/plans/phase6_wave2_plan.md. Run on a GPU pod from the repo root:

  python -m typed_decisions.rationales generate --out processed_data/phase6/all/rationales.jsonl
  python -m typed_decisions.rationales sample --path processed_data/phase6/all/rationales.jsonl   # quality check
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from typed_decisions.multitask_bench import load

# Training tasks closest to the reasoning types the student misses on held-out tasks (entailment,
# paraphrase, pronouns, physical/causal commonsense, reading comprehension).
RATIONALE_TASKS = ("mnli", "snli", "anli", "qqp", "paws", "wsc", "piqa", "swag", "csqa", "race")
TEACHER = "google/gemma-4-31B-it"

PROMPT = """Below is a question about a text, with its correct answer. In 1-3 sentences, explain the reasoning \
that makes this answer correct, pointing to the specific words or facts that decide it. Do not restate the \
question and do not start with "The answer is".

{state}

Question: {instruction}
Options: {options}
Correct answer: {answer}"""


def key(it):
    """Stable id for a training item (independent of file order)."""
    return hashlib.sha1("\x1f".join([it.task, it.state, it.instruction, *it.options]).encode()).hexdigest()[:16]


def select(train, per_task=700, seed=0):
    """Up to per_task items from each rationale task, deterministic."""
    rng, out = random.Random(seed), []
    for t in RATIONALE_TASKS:
        its = [x for x in train if x.task == t]
        rng.shuffle(its)
        out += its[:per_task]
    return out


def teacher_prompt(it):
    return PROMPT.format(state=it.state, instruction=it.instruction,
                         options=" | ".join(it.options), answer=it.options[it.label])


def clean(text, max_sentences=3):
    text = " ".join(text.strip().split())
    parts = [p for p in text.replace("! ", "!\n").replace("? ", "?\n").replace(". ", ".\n").split("\n") if p]
    return " ".join(parts[:max_sentences])


def load_rationales(path):
    return {r["key"]: r["rationale"] for r in map(json.loads, Path(path).read_text().splitlines()) if r["rationale"]}


def cmd_generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = select(load("train", args.data_dir), args.per_task, args.seed)
    out = Path(args.out)
    done = set(map(lambda l: json.loads(l)["key"], out.read_text().splitlines())) if out.exists() else set()
    todo = [it for it in items if key(it) not in done]
    print(f"{len(items)} selected, {len(done)} already done, {len(todo)} to generate", flush=True)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    # Longest prompts first: batches pad less, and an out-of-memory error shows up in the first batch.
    todo.sort(key=lambda it: -len(it.state))
    with out.open("a") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": teacher_prompt(it)}], tokenize=False,
                                             add_generation_prompt=True) for it in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            for it, g in zip(chunk, gen[:, enc["input_ids"].shape[1] :]):
                f.write(json.dumps({"key": key(it), "task": it.task, "label": it.label,
                                    "rationale": clean(tok.decode(g, skip_special_tokens=True))}) + "\n")
            f.flush()
            print(f"{min(i + args.batch, len(todo))}/{len(todo)}", flush=True)


def cmd_sample(args):
    items = {key(it): it for it in load("train", args.data_dir)}
    rows = [json.loads(l) for l in Path(args.path).read_text().splitlines()]
    rng = random.Random(args.seed)
    for r in rng.sample(rows, min(args.n, len(rows))):
        it = items[r["key"]]
        print(f"--- {it.task} | answer: {it.options[it.label]}\n{it.state[:400]}\nQ: {it.instruction}\nWHY: {r['rationale']}\n")
    lens = sorted(len(r["rationale"].split()) for r in rows)
    print(f"{len(rows)} rationales; words median {lens[len(lens) // 2]}, max {lens[-1]}; empty {sum(not r['rationale'] for r in rows)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--data_dir", default="processed_data/phase6/all")
    g.add_argument("--teacher", default=TEACHER)
    g.add_argument("--per_task", type=int, default=700)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--max_new_tokens", type=int, default=96)
    g.add_argument("--out", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--data_dir", default="processed_data/phase6/all")
    s.add_argument("--path", required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"generate": cmd_generate, "sample": cmd_sample}[args.cmd](args)


if __name__ == "__main__":
    main()
