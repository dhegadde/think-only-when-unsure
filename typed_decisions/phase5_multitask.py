#!/usr/bin/env python3
"""Phase 5: multitask typed-decision head on Gemma 4 E2B + LoRA, evaluated zero-shot on held-out decision
tasks. Protocol: goals/typed-decision-heads/plans/phase5_protocol.md. Run on a GPU pod from the repo root.

  train     LoRA + DecisionHead on the 6 training tasks (mixed batches; K varies per item)
  eval      one arm on the 10 held-out tasks: --arm cold (LM likelihood, no training) |
            --arm head --adapter DIR --head FILE (a trained adapter + head, applied as-is)

Row for option o: [bos] + gemma(state + "\\n\\nQuestion: " + instruction + "\\nAnswer:") + gemma(" " + o) + [eos].
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.schedules import make_lr_lambda
from typed_decisions.decision_head import DecisionHead
from typed_decisions.gemma_core import TARGETS, _peft_without_torchao, core, load_gemma, pad, readout
from typed_decisions.multitask_bench import HELDOUT_TASKS, load
from typed_decisions.rationales import key, load_rationales

def make_head(d, norm=False):
    """DecisionHead, optionally behind a LayerNorm without learned scale: Gemma's readout has a few huge
    activation dims (Phase 4); unnormalized, Gemma 4 E4B failed to learn at all (Phase 6 wave 1b)."""
    head = DecisionHead(d_model=d)
    return torch.nn.Sequential(torch.nn.LayerNorm(d, elementwise_affine=False), head) if norm else head


NEAR = {"imdb", "rte", "hellaswag"}
MEDIUM = {"arc_easy", "openbookqa"}
FAR = {"mrpc", "wic", "winogrande", "copa", "stsb"}


def item_rows(it, gtok):
    """Token rows for every option of one item, plus each option's [start, end) token span."""
    prefix = gtok(f"{it.state}\n\nQuestion: {it.instruction}\nAnswer:", add_special_tokens=True)["input_ids"]
    rows, spans = [], []
    for o in it.options:
        opt = gtok(" " + o, add_special_tokens=False)["input_ids"]
        rows.append(prefix + opt + [gtok.eos_token_id])
        spans.append((len(prefix), len(prefix) + len(opt)))
    return rows, spans


def rationale_row(it, text, gtok):
    """Phase 6 wave 2: [item prefix] + " <correct option>\nWhy:" + " <teacher rationale>" + [eos], and the index
    where the rationale starts (the next-token loss covers only the rationale and eos)."""
    prefix = gtok(f"{it.state}\n\nQuestion: {it.instruction}\nAnswer:", add_special_tokens=True)["input_ids"]
    ans = gtok(" " + it.options[it.label] + "\nWhy:", add_special_tokens=False)["input_ids"]
    rat = gtok(" " + text, add_special_tokens=False)["input_ids"] + [gtok.eos_token_id]
    return prefix + ans + rat, len(prefix) + len(ans)


def rationale_loss(model, row, start, device):
    """Mean next-token cross-entropy on the rationale tokens; lm_head (+ soft-capping) only on those positions."""
    c = core(model)
    cap = getattr(c.config.get_text_config(), "final_logit_softcapping", None)
    ids = torch.tensor([row], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = c.model(input_ids=ids).last_hidden_state
    lg = c.lm_head(h[0, start - 1 : -1]).float()
    if cap:
        lg = torch.tanh(lg / cap) * cap
    return F.cross_entropy(lg, ids[0, start:])


def split_scores(flat, ks):
    out, i = [], 0
    for k in ks:
        out.append(flat[i : i + k])
        i += k
    return out


def metrics(prob_lists, items):
    """accuracy, multi-class Brier, and for score tasks the mean abs error of the expected level."""
    acc, brier, mae = [], [], []
    for p, it in zip(prob_lists, items):
        p = torch.as_tensor(p, dtype=torch.float)
        y = torch.zeros_like(p)
        y[it.label] = 1
        acc.append(float(p.argmax().item() == it.label))
        brier.append(float(((p - y) ** 2).sum()))
        if it.kind == "score":
            mae.append(abs(float((p * torch.arange(len(p))).sum()) - it.label))
    out = {"n": len(items), "acc": sum(acc) / len(acc), "brier": sum(brier) / len(brier)}
    if mae:
        out["score_mae"] = sum(mae) / len(mae)
    return out


@torch.no_grad()
def head_probs(model, head, items, gtok, device, batch_items=8):
    model.eval()
    pad_id = gtok.pad_token_id if gtok.pad_token_id is not None else 0
    probs = []
    for i in range(0, len(items), batch_items):
        chunk = items[i : i + batch_items]
        rows = [item_rows(it, gtok)[0] for it in chunk]
        scores = head(readout(model, [r for rs in rows for r in rs], pad_id, device)).float().cpu()
        probs += [F.softmax(s, dim=0).tolist() for s in split_scores(scores, [len(r) for r in rows])]
    return probs


@torch.no_grad()
def head_probs_shared(model, head, items, gtok, device):
    """Same scores as head_probs, but the shared prefix (state + question) is encoded ONCE per item and its
    KV cache reused for the K option continuations (Phase 7: token cost of one-pass decisions)."""
    model.eval()
    m = core(model).model
    pad_id = gtok.pad_token_id if gtok.pad_token_id is not None else 0
    probs = []
    for it in items:
        rows, spans = item_rows(it, gtok)
        P, K = spans[0][0], len(rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cache = m(input_ids=torch.tensor([rows[0][:P]], device=device), use_cache=True).past_key_values
            cache.batch_repeat_interleave(K)
            ids, mask, lengths = pad([r[P:] for r in rows], pad_id, device)
            full_mask = torch.cat([torch.ones(K, P, dtype=mask.dtype, device=device), mask], 1)
            h = m(input_ids=ids, attention_mask=full_mask, past_key_values=cache, use_cache=True).last_hidden_state
        scores = head(h[torch.arange(K, device=device), lengths - 1].float()).float().cpu()
        probs.append(F.softmax(scores.reshape(-1), dim=0).tolist())
    return probs


@torch.no_grad()
def head_probs_shared_batched(model, head, items, gtok, device, batch_items=8):
    """head_probs_shared, batched across items: prefixes LEFT-padded into one batch (one prefix pass per item),
    the cache repeated Kmax times per item, options right-padded after it; explicit position ids so padding
    does not shift positions. Items with fewer options get empty dummy rows that are ignored."""
    model.eval()
    m = core(model).model
    pad_id = gtok.pad_token_id if gtok.pad_token_id is not None else 0
    probs = []
    for b in range(0, len(items), batch_items):
        built = [item_rows(it, gtok) for it in items[b : b + batch_items]]
        prefixes = [rows[0][: spans[0][0]] for rows, spans in built]
        Kmax, P = max(len(r) for r, _ in built), max(len(x) for x in prefixes)
        pids = torch.full((len(built), P), pad_id, dtype=torch.long)
        pmask = torch.zeros((len(built), P), dtype=torch.long)
        for i, x in enumerate(prefixes):
            pids[i, P - len(x) :] = torch.tensor(x)
            pmask[i, P - len(x) :] = 1
        pids, pmask = pids.to(device), pmask.to(device)
        ppos = (pmask.cumsum(1) - 1).clamp(min=0)
        tails = [r[len(x) :] for (rows, _), x in zip(built, prefixes) for r in rows + [[]] * (Kmax - len(rows))]
        L = max(1, max(len(t) for t in tails))
        tids = torch.full((len(tails), L), pad_id, dtype=torch.long, device=device)
        tmask = torch.zeros((len(tails), L), dtype=torch.long, device=device)
        for j, t in enumerate(tails):
            if t:
                tids[j, : len(t)] = torch.tensor(t, device=device)
                tmask[j, : len(t)] = 1
        plen = pmask.sum(1).repeat_interleave(Kmax)
        tpos = plen[:, None] + torch.arange(L, device=device)[None, :]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cache = m(input_ids=pids, attention_mask=pmask, position_ids=ppos, use_cache=True).past_key_values
            cache.batch_repeat_interleave(Kmax)
            full_mask = torch.cat([pmask.repeat_interleave(Kmax, 0), tmask], 1)
            h = m(input_ids=tids, attention_mask=full_mask, position_ids=tpos, past_key_values=cache,
                  use_cache=True).last_hidden_state
        last = (tmask.sum(1) - 1).clamp(min=0)
        scores = head(h[torch.arange(len(tails), device=device), last].float()).float().cpu().reshape(len(built), Kmax)
        probs += [F.softmax(scores[i, : len(rows)], dim=0).tolist() for i, (rows, _) in enumerate(built)]
    return probs


@torch.no_grad()
def cold_probs(model, items, gtok, device, batch_items=8):
    """Gemma's own summed log-likelihood of each option; lm_head (+ Gemma's final logit soft-capping)
    applied ONLY to the option positions, never to the full 262k-vocab x length tensor."""
    model.eval()
    c = core(model)
    cap = getattr(c.config.get_text_config(), "final_logit_softcapping", None)
    pad_id = gtok.pad_token_id if gtok.pad_token_id is not None else 0
    probs = []
    for i in range(0, len(items), batch_items):
        chunk = items[i : i + batch_items]
        built = [item_rows(it, gtok) for it in chunk]
        flat_rows = [r for rows, _ in built for r in rows]
        flat_spans = [s for _, spans in built for s in spans]
        ids, mask, _ = pad(flat_rows, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = c.model(input_ids=ids, attention_mask=mask).last_hidden_state
        lls = []
        for j, (a, b) in enumerate(flat_spans):
            lg = c.lm_head(h[j, a - 1 : b - 1]).float()
            if cap:
                lg = torch.tanh(lg / cap) * cap
            lls.append(F.log_softmax(lg, dim=-1).gather(1, ids[j, a:b].unsqueeze(1)).sum().item())
        probs += [F.softmax(torch.tensor(s), dim=0).tolist() for s in split_scores(lls, [len(r) for r, _ in built])]
    return probs


def cmd_train(args):
    _peft_without_torchao()
    from peft import LoraConfig, get_peft_model
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, gtok = load_gemma(device, model_name=args.model)
    if args.grad_ckpt:  # recompute activations in backward: needed to fit Gemma 4 E4B training in 32 GB
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS))
    d = core(model).config.get_text_config().hidden_size
    head = make_head(d, args.head_norm).to(device)
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
                             {"params": head.parameters(), "lr": args.head_lr}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda("wsd", args.steps, max(1, int(0.05 * args.steps)), decay_ratio=0.2))
    train, dev = load("train", args.data_dir), load("dev", args.data_dir)
    pad_id = gtok.pad_token_id if gtok.pad_token_id is not None else 0
    order, cur, curve, t0 = list(range(len(train))), 0, [], time.time()
    skipped, best_acc = [], -1.0
    rats, n_rat = (load_rationales(args.rationales) if args.rationales else {}), 0
    if rats:
        print(f"rationale loss on {sum(key(x) in rats for x in train)} of {len(train)} training items "
              f"(weight {args.rationale_weight})", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng.shuffle(order)
    for step in range(args.steps):
        model.train()
        if cur + args.batch_items > len(order):
            rng.shuffle(order)
            cur = 0
        chunk = [train[i] for i in order[cur : cur + args.batch_items]]
        cur += args.batch_items
        # One item per forward/backward, gradients accumulated over the batch: identical update to a joint
        # batch, but peak memory is one item (4 long items x 5 options ran a 32 GB GPU out of memory).
        opt.zero_grad()
        loss_sum, finite = 0.0, True
        for it in chunk:
            rows = item_rows(it, gtok)[0]
            s = head(readout(model, rows, pad_id, device))
            loss = F.cross_entropy(s.unsqueeze(0), torch.tensor([it.label], device=device)) / len(chunk)
            if rats and key(it) in rats:
                row, start = rationale_row(it, rats[key(it)], gtok)
                loss = loss + args.rationale_weight * rationale_loss(model, row, start, device) / len(chunk)
                n_rat += 1
            if not torch.isfinite(loss):  # stability guard (Phase 6 wave 2): E4B blew up at ~step 1,250
                finite = False
                break
            loss.backward()
            loss_sum += loss.item()
        loss = torch.tensor(loss_sum)
        if finite:
            gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            finite = bool(torch.isfinite(gnorm))
        if finite:
            opt.step()
        else:
            opt.zero_grad()
            skipped.append(step + 1)
            print(f"step {step + 1}: non-finite loss/grad, update skipped ({len(skipped)} so far)", flush=True)
        sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            m = metrics(head_probs(model, head, dev, gtok, device), dev)
            curve.append({"step": step + 1, "dev_acc": m["acc"], "dev_brier": m["brier"], "loss": loss.item(),
                          "skipped": len(skipped), "seconds": time.time() - t0})
            print(curve[-1], flush=True)
            if m["acc"] > best_acc:  # keep the best-dev checkpoint too (reported alongside end-of-schedule)
                best_acc = m["acc"]
                model.save_pretrained(out / "best" / "adapter")
                torch.save(head.state_dict(), out / "best" / "head.pt")
                (out / "best" / "step.json").write_text(json.dumps({"step": step + 1, "dev_acc": best_acc}))
    model.save_pretrained(out / "adapter")
    torch.save(head.state_dict(), out / "head.pt")
    dev_by_task = {t: metrics(p, its) for t, its, p in
                   [(t, [x for x in dev if x.task == t], None) for t in sorted({x.task for x in dev})]
                   for p in [head_probs(model, head, its, gtok, device)]}
    (out / "result.json").write_text(json.dumps({"seed": args.seed, "steps": args.steps, "curve": curve,
                                                 "skipped_steps": skipped, "best_dev_acc": best_acc,
                                                 "rationales": args.rationales, "rationale_weight": args.rationale_weight,
                                                 "rationale_items_seen": n_rat,
                                                 "dev_by_task": dev_by_task, "seconds": time.time() - t0}, indent=2))


def cmd_eval(args):
    device = torch.device("cuda")
    model, gtok = load_gemma(device, adapter=args.adapter, model_name=args.model)
    head = None
    if args.arm == "head":
        head = make_head(core(model).config.get_text_config().hidden_size, args.head_norm).to(device)
        head.load_state_dict(torch.load(args.head, map_location=device))
        head.eval()
    res, per_item, secs = {}, {}, {}
    score = (lambda its: cold_probs(model, its, gtok, device)) if args.arm == "cold" else \
        (lambda its: head_probs_shared_batched(model, head, its, gtok, device)) if args.shared_prefix == "batched" else \
        (lambda its: head_probs_shared(model, head, its, gtok, device)) if args.shared_prefix else \
        (lambda its: head_probs(model, head, its, gtok, device))
    if args.split == "dev":  # in-distribution dev items (training tasks): Phase 7 cascade threshold selection
        items = load("dev")
        per_item["dev"] = score(items)
        res["dev"] = metrics(per_item["dev"], items)
        Path(args.save_probs).write_text(json.dumps(per_item))
        print(f"{args.name} dev: acc {res['dev']['acc']:.3f}", flush=True)
        return
    for t in HELDOUT_TASKS:
        items = load(f"test_{t}")
        torch.cuda.synchronize()
        t0 = time.time()
        p = score(items)
        torch.cuda.synchronize()
        secs[t] = time.time() - t0
        res[t] = metrics(p, items)
        per_item[t] = p
        print(f"{args.name} {t}: acc {res[t]['acc']:.3f} brier {res[t]['brier']:.3f}", flush=True)
    for g, ts in (("near", NEAR), ("medium", MEDIUM), ("far", FAR), ("all", set(HELDOUT_TASKS))):
        res[f"avg_{g}"] = {k: sum(res[t][k] for t in ts) / len(ts) for k in ("acc", "brier")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"arm": args.arm, "name": args.name, "adapter": args.adapter, "results": res,
                                          "shared_prefix": args.shared_prefix, "seconds": secs}, indent=2))
    if args.save_probs:  # per-item option probabilities, in test-file order (Phase 7 cascade analysis)
        Path(args.save_probs).write_text(json.dumps(per_item))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_items", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--data_dir", default="processed_data/phase5", help="training mix (Phase 6: processed_data/phase6/<mix>)")
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (memory for larger models)")
    p.add_argument("--head_norm", action="store_true", help="LayerNorm (no affine) on the readout before the head")
    p.add_argument("--rationales", default=None, help="teacher rationales jsonl (typed_decisions.rationales)")
    p.add_argument("--rationale_weight", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--arm", choices=("cold", "head"), required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--head", default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--head_norm", action="store_true", help="must match how the head was trained")
    p.add_argument("--save_probs", default=None, help="also write per-item probabilities (json) here")
    p.add_argument("--shared_prefix", nargs="?", const="single", default=None, choices=("single", "batched"),
                   help="head arm: encode the shared prefix once (KV cache); 'batched' = across items too")
    p.add_argument("--split", choices=("test", "dev"), default="test", help="dev: in-distribution dev items only")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
