"""Phase 5 benchmark: real decision tasks in one Jev-style shape (plans/phase5_protocol.md).

Every example is a DecisionItem: `state` (input text), `instruction` (the question in words), `options`
(candidate answers; yes/no -> ["Yes", "No"]; scores -> one described level per option), `label` (index of
the correct option) and `kind` ("choice" | "noul" | "score"). Training tasks draw from train splits; held-out
tasks draw only from validation/test splits and never appear in training.

    python -m typed_decisions.multitask_bench   # writes processed_data/phase5/*.jsonl
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

OUT = Path("processed_data/phase5")
SEED = 500
N_TRAIN, N_DEV, N_TEST = 2000, 150, 500
MAX_CHARS = 1500  # long inputs (passages, reviews) are cut to keep rows short
YES_NO = ["Yes", "No"]
TRAIN_TASKS = ["boolq", "sst2", "mnli", "csqa", "yelp", "coherence"]
HELDOUT_TASKS = ["imdb", "rte", "hellaswag", "arc_easy", "openbookqa", "mrpc", "wic", "winogrande", "copa", "stsb"]


@dataclass
class DecisionItem:
    task: str
    kind: str
    state: str
    instruction: str
    options: List[str]
    label: int


def _cut(t: str) -> str:
    t = " ".join(t.split())
    return t if len(t) <= MAX_CHARS else t[:MAX_CHARS].rsplit(" ", 1)[0] + " ..."


def _sample(ds, n, rng):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [ds[i] for i in idx[:n]]


def _pair(a, b, na="Sentence 1", nb="Sentence 2"):
    return f"{na}: {_cut(a)}\n{nb}: {_cut(b)}"


def _yn(task, state, instruction, is_yes):
    return DecisionItem(task, "noul", state, instruction, YES_NO, 0 if is_yes else 1)


# ---------------------------------------------------------------- converters (one row -> DecisionItem)
def conv(task, r):
    if task == "boolq":
        return _yn(task, _cut(r["passage"]), r["question"].strip().rstrip("?").capitalize() + "?", bool(r["answer"]))
    if task in ("sst2", "imdb"):
        text = r["sentence"] if task == "sst2" else r["text"]
        return _yn(task, _cut(text.replace("<br />", " ")), "Is the sentiment of this text positive?", r["label"] == 1)
    if task == "mnli":
        return DecisionItem(task, "choice", _pair(r["premise"], r["hypothesis"], "Premise", "Hypothesis"),
                            "How does the hypothesis relate to the premise?",
                            ["The premise entails the hypothesis.", "The hypothesis may or may not be true given the premise.",
                             "The premise contradicts the hypothesis."], int(r["label"]))
    if task == "csqa":
        return DecisionItem(task, "choice", r["question"], "Which answer is correct?", list(r["choices"]["text"]),
                            list(r["choices"]["label"]).index(r["answerKey"]))
    if task == "yelp":
        return DecisionItem(task, "score", _cut(r["text"]), "How many stars did the reviewer give?",
                            ["1 star: very negative", "2 stars: negative", "3 stars: mixed or neutral",
                             "4 stars: positive", "5 stars: very positive"], int(r["label"]))
    if task == "rte":  # glue rte: 0 = entailment
        return _yn(task, _pair(r["sentence1"], r["sentence2"], "Premise", "Hypothesis"),
                   "Does the premise entail the hypothesis?", r["label"] == 0)
    if task == "hellaswag":
        return DecisionItem(task, "choice", r["ctx"], "Which ending continues the text best?",
                            [e.strip() for e in r["endings"]], int(r["label"]))
    if task in ("arc_easy", "openbookqa"):
        q = r["question"] if task == "arc_easy" else r["question_stem"]
        return DecisionItem(task, "choice", q, "Which answer is correct?", list(r["choices"]["text"]),
                            list(r["choices"]["label"]).index(r["answerKey"]))
    if task == "mrpc":
        return _yn(task, _pair(r["sentence1"], r["sentence2"]), "Do the two sentences mean the same thing?", r["label"] == 1)
    if task == "wic":
        return _yn(task, f"Word: {r['word']}\n" + _pair(r["sentence1"], r["sentence2"]),
                   "Is the word used with the same meaning in both sentences?", r["label"] == 1)
    if task == "winogrande":
        return DecisionItem(task, "choice", r["sentence"], "Which option correctly fills the blank (_)?",
                            [r["option1"], r["option2"]], int(r["answer"]) - 1)
    if task == "copa":
        q = "What was the cause?" if r["question"] == "cause" else "What happened as a result?"
        return DecisionItem(task, "choice", r["premise"], q, [r["choice1"], r["choice2"]], int(r["label"]))
    if task == "stsb":
        levels = ["0: completely different meaning", "1: not equivalent, but on the same topic",
                  "2: not equivalent, but share some details", "3: roughly equivalent, some important details differ",
                  "4: mostly equivalent, only minor details differ", "5: completely equivalent"]
        return DecisionItem(task, "score", _pair(r["sentence1"], r["sentence2"]),
                            "How similar in meaning are the two sentences?", levels, int(round(float(r["label"]))))
    # ---- Phase 6 additions (training only) ----
    nli3 = ["The premise entails the hypothesis.", "The hypothesis may or may not be true given the premise.",
            "The premise contradicts the hypothesis."]
    if task in ("snli", "anli"):
        return DecisionItem(task, "choice", _pair(r["premise"], r["hypothesis"], "Premise", "Hypothesis"),
                            "How does the hypothesis relate to the premise?", nli3, int(r["label"]))
    if task == "qqp":
        return _yn(task, _pair(r["question1"], r["question2"], "Question 1", "Question 2"),
                   "Do the two questions ask the same thing?", r["label"] == 1)
    if task == "paws":
        return _yn(task, _pair(r["sentence1"], r["sentence2"]), "Do the two sentences mean the same thing?", r["label"] == 1)
    if task == "swag":
        return DecisionItem(task, "choice", f"{r['sent1']} {r['sent2']}", "Which ending continues the text best?",
                            [r[f"ending{i}"] for i in range(4)], int(r["label"]))
    if task == "race":
        return DecisionItem(task, "choice", _cut(r["article"]), r["question"], list(r["options"]), "ABCD".index(r["answer"]))
    if task == "sciq":
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = list(range(4))
        random.Random(r["question"]).shuffle(order)
        return DecisionItem(task, "choice", r["question"], "Which answer is correct?", [opts[i] for i in order], order.index(0))
    if task == "qasc":
        return DecisionItem(task, "choice", r["question"], "Which answer is correct?", list(r["choices"]["text"]),
                            list(r["choices"]["label"]).index(r["answerKey"]))
    if task == "multirc":
        return _yn(task, _cut(r["paragraph"]), f"{r['question']} Is this answer correct: {r['answer']}", r["label"] == 1)
    if task == "wsc":
        return _yn(task, r["text"], f"In this text, does \"{r['span2_text']}\" refer to \"{r['span1_text']}\"?", r["label"] == 1)
    if task == "emotion":
        return DecisionItem(task, "choice", r["text"], "Which emotion does the text express?",
                            ["sadness", "joy", "love", "anger", "fear", "surprise"], int(r["label"]))
    if task == "ag_news":
        return DecisionItem(task, "choice", _cut(r["text"]), "What is the topic of this news article?",
                            ["World", "Sports", "Business", "Science and technology"], int(r["label"]))
    if task == "piqa":
        return DecisionItem(task, "choice", f"Goal: {r['goal']}", "Which solution achieves the goal?",
                            [r["sol1"], r["sol2"]], int(r["label"]))
    raise ValueError(task)


SOURCES = {  # task -> (dataset, config, split used)
    "boolq": ("google/boolq", None, "train"), "sst2": ("stanfordnlp/sst2", None, "train"),
    "mnli": ("nyu-mll/multi_nli", None, "train"), "csqa": ("tau/commonsense_qa", None, "train"),
    "yelp": ("Yelp/yelp_review_full", None, "train"),
    "imdb": ("stanfordnlp/imdb", None, "test"), "rte": ("nyu-mll/glue", "rte", "validation"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"), "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "openbookqa": ("allenai/openbookqa", "main", "test"), "mrpc": ("nyu-mll/glue", "mrpc", "validation"),
    "wic": ("aps/super_glue", "wic", "validation"), "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    "copa": ("aps/super_glue", "copa", "validation"), "stsb": ("nyu-mll/glue", "stsb", "validation"),
    # Phase 6 training additions (train splits only)
    "snli": ("stanfordnlp/snli", None, "train"), "anli": ("facebook/anli", None, "train_r1"),
    "qqp": ("nyu-mll/glue", "qqp", "train"), "paws": ("google-research-datasets/paws", "labeled_final", "train"),
    "swag": ("allenai/swag", "regular", "train"), "race": ("ehovy/race", "all", "train"),
    "sciq": ("allenai/sciq", None, "train"), "qasc": ("allenai/qasc", None, "train"),
    "multirc": ("aps/super_glue", "multirc", "train"), "wsc": ("aps/super_glue", "wsc.fixed", "train"),
    "emotion": ("dair-ai/emotion", None, "train"), "ag_news": ("fancyzhx/ag_news", None, "train"),
    "piqa": ("baber/piqa", None, "train"),
}
PHASE6_MIXES = {  # plans/phase6_protocol.md
    "far": TRAIN_TASKS + ["emotion", "ag_news", "race", "multirc", "piqa"],
    "all": TRAIN_TASKS + ["emotion", "ag_news", "race", "multirc", "piqa",
                          "snli", "anli", "qqp", "paws", "swag", "sciq", "qasc", "wsc"],
}


def build_mix(mix: str) -> Dict[str, List[DecisionItem]]:
    """Training + dev items for a Phase 6 mix (held-out tests stay in processed_data/phase5)."""
    from datasets import load_dataset
    rng = random.Random(SEED)
    out: Dict[str, List[DecisionItem]] = {"train": [], "dev": []}
    for task in PHASE6_MIXES[mix]:
        if task == "coherence":
            out["train"] += coherence_items("train", N_TRAIN, rng)
            out["dev"] += coherence_items("dev", N_DEV, rng)
            continue
        name, cfg, split = SOURCES[task]
        rows = [r for r in _sample(load_dataset(name, cfg, split=split), N_TRAIN + N_DEV + 500, rng)
                if not (task in ("mnli", "snli", "anli") and r["label"] not in (0, 1, 2))]
        items = [conv(task, r) for r in rows][: N_TRAIN + N_DEV]
        n_dev = min(N_DEV, len(items) // 10)
        out["train"] += items[: len(items) - n_dev] if len(items) < N_TRAIN + N_DEV else items[:N_TRAIN]
        out["dev"] += items[-n_dev:] if len(items) < N_TRAIN + N_DEV else items[N_TRAIN:]
    return out


def coherence_items(split: str, n: int, rng) -> List[DecisionItem]:
    """Our coherence benchmark (shard0 train/dev split only) as text."""
    from configs.dataset_config import DataConfig
    from data.loader import setup_tokenizer
    from typed_decisions.phase2_joint_train import build_shared_benchmark
    tok = setup_tokenizer(DataConfig(dataset_path="processed_data/shard0_30M", cache_dir="./hf_cache"))
    exs = build_shared_benchmark()[split]
    rng.shuffle(exs)
    return [DecisionItem("coherence", "choice", tok.decode(e.state, skip_special_tokens=True),
                         "Which passage continues the text?",
                         [tok.decode(c, skip_special_tokens=True) for c in e.candidates], e.true_index) for e in exs[:n]]


def build() -> Dict[str, List[DecisionItem]]:
    from datasets import load_dataset
    rng = random.Random(SEED)
    out: Dict[str, List[DecisionItem]] = {"train": [], "dev": []}
    for task in TRAIN_TASKS:
        if task == "coherence":
            out["train"] += coherence_items("train", N_TRAIN, rng)
            out["dev"] += coherence_items("dev", N_DEV, rng)
            continue
        name, cfg, split = SOURCES[task]
        rows = [r for r in _sample(load_dataset(name, cfg, split=split), N_TRAIN + N_DEV + 500, rng)
                if not (task == "mnli" and r["label"] not in (0, 1, 2))]
        items = [conv(task, r) for r in rows][: N_TRAIN + N_DEV]
        out["train"] += items[:N_TRAIN]
        out["dev"] += items[N_TRAIN:]
    for task in HELDOUT_TASKS:
        name, cfg, split = SOURCES[task]
        out[f"test_{task}"] = [conv(task, r) for r in _sample(load_dataset(name, cfg, split=split), N_TEST, rng)]
    return out


def save(data: Dict[str, List[DecisionItem]], out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    for name, items in data.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(asdict(it)) + "\n")


def load(name: str, data_dir: Path = OUT) -> List[DecisionItem]:
    with open(Path(data_dir) / f"{name}.jsonl") as f:
        return [DecisionItem(**json.loads(line)) for line in f]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:  # python -m typed_decisions.multitask_bench far|all
        data = build_mix(sys.argv[1])
        save(data, Path("processed_data/phase6") / sys.argv[1])
    else:
        data = build()
        save(data)
    for k, v in data.items():
        print(k, len(v))
