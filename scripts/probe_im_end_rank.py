"""Probe <|im_end|> logit rank at answer boundary across model variants.

Tests whether LoRA all-linear can shift hidden states sufficiently toward
<|im_end|> without training lm_head, or if lm_head must be unfrozen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    # Simple single-word answers
    ("Reply with exactly BLUE in uppercase and nothing else.", "BLUE"),
    ("Output DONE and stop immediately. Do not explain.", "DONE"),
    ('Return only a JSON object with key "name" and value "Alice".', '{"name":"Alice"}'),
    ("Classify 'A triangle has three sides' as true or false. Output only one word.", "true"),
]

CHAT_TEMPLATES = {
    # Training template (has {% generation %} markers, used by both LoRA adapters)
    "training": None,
    # Eval template (no generation markers, used for base model)
    "eval": None,
}


def _load_chat_templates():
    """Load training and eval chat templates."""
    from src.training.chat_template import (
        QWEN3_BASE_CHAT_TEMPLATE,
        QWEN3_TRAINING_CHAT_TEMPLATE,
    )
    CHAT_TEMPLATES["eval"] = QWEN3_BASE_CHAT_TEMPLATE
    CHAT_TEMPLATES["training"] = QWEN3_TRAINING_CHAT_TEMPLATE


def _set_template(tokenizer, template_type: str):
    tokenizer.chat_template = CHAT_TEMPLATES[template_type]


def probe(model, tokenizer, prompt_text: str, answer: str) -> dict:
    """Forward pass and check <|im_end|> logit stats at answer boundary."""
    # Format prompt: user message + assistant prefix
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # Assemble full input: prompt + answer (but NOT im_end)
    full_text = prompt + answer
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    # Logits at the last position
    logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Sort logits descending
    sorted_logits, sorted_indices = logits.sort(descending=True)
    rank = (sorted_indices == im_end_id).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
    prob = torch.softmax(logits.float(), dim=-1)
    im_end_prob = prob[im_end_id].item()
    top1_id = sorted_indices[0].item()
    top1_token = tokenizer.decode([top1_id])
    top1_prob = prob[top1_id].item()

    # Also check top-5
    top5 = [(tokenizer.decode([sorted_indices[i].item()]), prob[sorted_indices[i]].item())
            for i in range(5)]

    return {
        "im_end_rank": rank,
        "im_end_prob": im_end_prob,
        "im_end_logit": logits[im_end_id].item(),
        "top1_token": repr(top1_token),
        "top1_prob": top1_prob,
        "top5": top5,
        "vocab_size": logits.shape[0],
    }


def main():
    _load_chat_templates()

    base_model_id = "models/Qwen3-1.7B-Base"
    formal_adapter = "outputs/stage1_no_robots_qwen3_1_7b_local/formal/adapter"
    stop_overfit_adapter = "outputs/stop_overfit/adapter"

    # ── Load tokenizer once ──
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    variants = []

    # 1. Base model
    print("Loading Base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    base.eval()
    _set_template(tokenizer, "eval")
    variants.append(("Base (no LoRA)", base, tokenizer))

    # 2. LoRA all-linear (formal adapter)
    print("Loading LoRA all-linear (formal)...")
    from peft import PeftModel
    base2 = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    lora_alllinear = PeftModel.from_pretrained(base2, formal_adapter, is_trainable=False)
    lora_alllinear.eval()
    # Reload tokenizer with adapter's template
    tok2 = AutoTokenizer.from_pretrained(formal_adapter)
    _set_template(tok2, "training")
    variants.append(("LoRA all-linear (formal)", lora_alllinear, tok2))

    # 3. LoRA all-linear + lm_head (stop overfit)
    print("Loading LoRA all-linear + lm_head (stop_overfit)...")
    base3 = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    lora_lmhead = PeftModel.from_pretrained(base3, stop_overfit_adapter, is_trainable=False)
    lora_lmhead.eval()
    tok3 = AutoTokenizer.from_pretrained(stop_overfit_adapter)
    _set_template(tok3, "training")
    variants.append(("LoRA + lm_head (stop_overfit)", lora_lmhead, tok3))

    # ── Run probes ──
    print("\n" + "=" * 80)
    print("PROBE: <|im_end|> logit rank at answer boundary")
    print("Lower rank = model more confident that next token is <|im_end|>")
    print("=" * 80)

    for prompt_text, answer in PROMPTS:
        print(f"\n{'─' * 70}")
        print(f"Prompt: {prompt_text[:80]}")
        print(f"Expected answer: {answer}")
        print(f"{'─' * 70}")
        print(f"{'Model':<35} {'Rank':>6}  {'Prob':>10}  {'Top-1 Token':>20}")
        print(f"{'─' * 70}")

        for name, model, tok in variants:
            result = probe(model, tok, prompt_text, answer)
            rank = result["im_end_rank"]
            prob = result["im_end_prob"]
            print(f"{name:<35} {rank:>6}  {prob:>10.6f}  {result['top1_token']:>20}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  - If LoRA all-linear rank >> 1: hidden state shift is insufficient")
    print("    without training lm_head")
    print("  - If LoRA all-linear rank ~ Base rank: LoRA can't shift at all")
    print("  - If LoRA + lm_head rank ~ 1: lm_head training is necessary")
    print("=" * 80)


if __name__ == "__main__":
    main()
