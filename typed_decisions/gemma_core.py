#!/usr/bin/env python3
"""Gemma 4 helpers shared by the decision-head code: load a text-only Gemma 4 (optionally with a LoRA adapter),
final hidden states without the 262k-vocab logits, padding and the end-of-row readout. No dependency on the
rest of the lab kit."""
import torch

MODEL = "google/gemma-4-E2B"
TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def _peft_without_torchao():
    """peft probes torchao to support torchao-QUANTIZED layers (not used here) and refuses torchao < 0.16,
    while this lab pins torchao 0.11 for torchtune. Tell peft torchao is absent, for this process only."""
    import peft.import_utils as iu
    import peft.tuners.lora.torchao as lt
    iu.is_torchao_available = lambda: False
    lt.is_torchao_available = lambda: False


def load_gemma(device, adapter=None, model_name=MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    for part in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):  # text only
        if hasattr(model.model, part):
            setattr(model.model, part, None)
    model = model.to(device)
    if adapter is not None:
        _peft_without_torchao()
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def core(model):
    """The Gemma4ForConditionalGeneration object, whether or not it is wrapped by peft."""
    return model.base_model.model if hasattr(model, "peft_config") else model


def hidden(model, ids, mask):
    """Last-layer hidden states without computing the 262k-vocab logits."""
    return core(model).model(input_ids=ids, attention_mask=mask).last_hidden_state


def pad(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device), torch.tensor([len(s) for s in seqs], device=device)


def readout(model, seqs, pad_id, device):
    ids, mask, lengths = pad(seqs, pad_id, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = hidden(model, ids, mask)
    return h[torch.arange(len(seqs), device=device), lengths - 1].float()
