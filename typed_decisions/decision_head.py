"""Typed decision head on a MinimalLLM backbone (goals/typed-decision-heads, Phase 0).

Phase 0 scope is correctness, not speed. Each (state, question, candidate) triple is packed into
its own token sequence [state_tokens, candidate_tokens, READOUT] and run through the backbone as
an ordinary batch row (right-padded, standard causal attention -- no change to models/layers.py).
Because attention is causal and padding is always placed after the true content, the hidden state
at a row's true last token never depends on its padding (test_decision_head.py checks this
directly). There is no shared-prefix reuse across candidates yet -- that is NanoJev's
tree-attention trick (see nanojev-study/repo/scripts/check_qwen_tree.py), deferred to a later,
explicit efficiency phase once this simple version is known to work.

READOUT marker: we reuse the tokenizer's EOS id (0 for the SmolLM2 tokenizer this project uses)
rather than reserving a new vocab entry, matching the repo's existing convention of pad_token =
eos_token. We rely on POSITION (`lengths - 1`), not token identity, to find the readout hidden
state, so reusing EOS for both padding-fill and the readout marker is unambiguous.

Primitives, matching Jev/NanoJev's naming (docs.typesafe.ai/primitives.md):
  - Choice: log-softmax over a FIXED number of candidates per question. (Jev/NanoJev support
    2-255 *dynamic* candidates; fixed-K is a Phase 0 simplification, noted in MISSION.md.)
  - Boolean (Jev calls this Noul): an independent sigmoid per candidate, "is this the right one".
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionHead(nn.Module):
    """Scores one packed (state, question, candidate) sequence from its readout hidden state."""

    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, readout_hidden: torch.Tensor) -> torch.Tensor:
        """readout_hidden: (N, d_model) -> (N,) raw (pre-softmax/sigmoid) scores."""
        return self.net(readout_hidden).squeeze(-1)


def pack_batch(token_rows: Sequence[Sequence[int]], pad_id: int, max_len: int | None = None):
    """Right-pad variable-length token rows into one batch tensor.

    Returns (tokens: LongTensor (N, L), lengths: LongTensor (N,) — the true, unpadded length of
    each row, used to find the readout position; never inferred from `pad_id`, since a real
    sequence may legitimately contain that id (it's the tokenizer's EOS)).
    """
    if not token_rows:
        raise ValueError("pack_batch: no rows given")
    lengths = torch.tensor([len(r) for r in token_rows], dtype=torch.long)
    longest = int(lengths.max().item())
    L = max_len or longest
    if longest > L:
        raise ValueError(f"a row of length {longest} exceeds max_len={L}")
    tokens = torch.full((len(token_rows), L), pad_id, dtype=torch.long)
    for i, row in enumerate(token_rows):
        t = row if torch.is_tensor(row) else torch.tensor(list(row), dtype=torch.long)
        tokens[i, : t.numel()] = t
    return tokens, lengths


def gather_readout(hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """hidden_states: (N, seq_len, d_model), lengths: (N,) -> (N, d_model) at index length-1."""
    idx = (lengths - 1).clamp(min=0)
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), idx]


def score_rows(model, head: DecisionHead, token_rows: Sequence[Sequence[int]], pad_id: int,
              max_len: int | None = None) -> torch.Tensor:
    """End to end: pack -> backbone.encode -> gather readout -> head. Returns (N,) raw scores."""
    device = next(model.parameters()).device
    tokens, lengths = pack_batch(token_rows, pad_id, max_len)
    tokens, lengths = tokens.to(device), lengths.to(device)
    hidden = model.encode(tokens)
    readout = gather_readout(hidden, lengths)
    return head(readout)


def choice_log_probs(scores: torch.Tensor, num_candidates: int) -> torch.Tensor:
    """scores: (num_questions * num_candidates,) flat, grouped contiguously, K per question.
    Returns (num_questions, K) log-softmax over candidates within each question.
    """
    if scores.numel() % num_candidates != 0:
        raise ValueError(f"{scores.numel()} scores not divisible by num_candidates={num_candidates}")
    return F.log_softmax(scores.view(-1, num_candidates), dim=-1)


def make_choice_rows(state: Sequence[int], candidates: Sequence[Sequence[int]], readout_id: int) -> List[List[int]]:
    """One packed row per candidate: state ++ candidate ++ [readout_id]."""
    return [list(state) + list(c) + [readout_id] for c in candidates]


def choice_brier_score(log_probs: torch.Tensor, true_index: torch.Tensor) -> torch.Tensor:
    """Multi-class Brier score per question: sum_k (p_k - y_k)^2, y_k=1 for the true candidate.
    log_probs: (N, K), true_index: (N,) -> (N,) scores, lower is better (0 = perfect, certain).
    """
    probs = log_probs.exp()
    y = torch.zeros_like(probs).scatter_(1, true_index.unsqueeze(1), 1.0)
    return ((probs - y) ** 2).sum(dim=-1)
