"""Standalone overfit script for testing short-answer stop behavior.

This bypasses the full training pipeline to directly test whether training on
short-answer instruction-following data teaches the model to emit <|im_end|>.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import SFTConfig

from src.training.chat_template import configure_training_chat_template, resolve_stop_token_ids
from src.common.config import load_yaml_config


def _make_formatting_func(tokenizer):
    """Create a formatting function that applies the training chat template."""
    def fmt(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return text
    return fmt


def _find_peft_config(config):
    """Create PEFT/LoRA config from training config."""
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        modules_to_save=["lm_head"],  # CRITICAL: train lm_head to learn stop token
    )


def _do_generate(model, tokenizer, messages_list, max_new_tokens=128):
    """Quick generation helper for verification."""
    from src.training.chat_template import render_generation_prompt

    results = []
    stop_ids = resolve_stop_token_ids(tokenizer)
    for messages in messages_list:
        prompt = render_generation_prompt(tokenizer, list(messages))
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=list(stop_ids.values()),
            )
        generated_ids = output[0][prompt_len:].tolist()
        stop_by_id = {v: k for k, v in stop_ids.items()}
        stop_reason = "length"
        for idx, tid in enumerate(generated_ids):
            if tid in stop_by_id:
                generated_ids = generated_ids[:idx + 1]
                stop_reason = stop_by_id[tid]
                break
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append({
            "text": text,
            "stop_reason": stop_reason,
            "generated_tokens": len(generated_ids),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_no_robots_qwen3_1_7b_local.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/instruction/stop_overfit"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stop_overfit"))
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--test-only", action="store_true", help="Test existing adapter without training")
    parser.add_argument("--test-prompts", type=int, default=5, help="Number of test prompts to run")
    args = parser.parse_args()

    if args.test_only:
        _test_standalone(args)
        return

    _train_standalone(args)


def _train_standalone(args):
    config = load_yaml_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer with training template ──
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    tokenizer.padding_side = "right"
    configure_training_chat_template(tokenizer)

    # ── Load base model ──
    from transformers import AutoModelForCausalLM
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"],
        dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.config.use_cache = False

    # ── Apply LoRA ──
    peft_config = _find_peft_config(config)
    print(f"Applying LoRA (r={peft_config.r}, alpha={peft_config.lora_alpha})...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Load dataset ──
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "validation.jsonl"),
        },
    )
    print(f"Train: {len(dataset['train'])} samples, Valid: {len(dataset['validation'])} samples")

    # ── Training args ──
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=1,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=int(args.num_steps * 0.03),
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=5,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=False,
        assistant_only_loss=True,  # CRITICAL: use {% generation %} markers for loss masking
        seed=42,
    )

    # ── Trainer ──
    # Use formatting_func with training chat template (has {% generation %} markers)
    # SFTTrainer auto-handles assistant-only loss masking via these markers
    from trl import SFTTrainer

    tokenizer.model_max_length = args.max_length

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # ── Train ──
    print(f"\nTraining for {args.num_steps} steps on {len(dataset['train'])} samples...")
    print(f"Each sample has short answer (1-20 words). Goal: show <|im_end|> is learnable.")
    trainer.train()

    # ── Save adapter ──
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    # Also save the training chat template
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nAdapter saved to {adapter_dir}")

    # ── Quick verification: test on training samples ──
    print("\n" + "=" * 60)
    print("Verification: Testing on training samples...")
    print("=" * 60)
    model.eval()
    test_samples = dataset["train"].select(range(min(8, len(dataset["train"]))))
    results = _do_generate(
        model, tokenizer,
        [s["messages"] for s in test_samples],
        max_new_tokens=64,
    )
    stopped = sum(1 for r in results if r["stop_reason"] != "length")
    for i, (s, r) in enumerate(zip(test_samples, results)):
        assistant = s["messages"][-1]["content"]
        match = "✓" if r["text"].strip() == assistant.strip() or assistant.strip() in r["text"].strip() else "✗"
        print(f"  [{match}] Expected: {assistant[:60]}")
        print(f"       Got:      {r['text'][:80]}")
        print(f"       Stop: {r['stop_reason']}, tokens: {r['generated_tokens']}")
    print(f"\n  Stop rate: {stopped}/{len(results)} ({stopped/len(results)*100:.0f}%)")

    # ── Save metrics ──
    metrics = {
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": len(dataset["train"]),
        "valid_samples": len(dataset["validation"]),
        "stop_verification": {
            "total": len(results),
            "stopped": stopped,
            "stop_rate": stopped / len(results) if results else 0,
        },
    }
    with open(output_dir / "overfit_result.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {output_dir / 'overfit_result.json'}")


def _test_standalone(args):
    """Test an existing adapter without training."""
    config = load_yaml_config(args.config)
    adapter_dir = Path(args.output_dir) / "adapter"
    if not adapter_dir.is_dir():
        print(f"ERROR: adapter not found at {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    from src.training.model_loader import load_lora_model, load_tokenizer
    from src.training.chat_template import render_generation_prompt

    tokenizer = load_tokenizer(config, for_training=False)
    # Reload with adapter's tokenizer if available
    if (adapter_dir / "tokenizer_config.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))

    model = load_lora_model(config, str(adapter_dir), is_trainable=False).eval()

    # Load some test prompts from the dataset
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.data_dir / "train.jsonl")},
    )
    test_samples = dataset["train"].select(range(min(args.test_prompts, len(dataset["train"]))))

    stop_ids = resolve_stop_token_ids(tokenizer)
    print(f"Testing {len(test_samples)} prompts on adapter from {adapter_dir}")
    print(f"Stop token IDs: {stop_ids}")
    print()

    for s in test_samples:
        msgs = s["messages"]
        prompt = render_generation_prompt(tokenizer, list(msgs))
        print(f"User: {msgs[-1]['content'][:100]}")
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=list(stop_ids.values()),
            )
        gen_ids = output[0][prompt_len:].tolist()
        stop_by_id = {v: k for k, v in stop_ids.items()}
        stop_reason = "length"
        for idx, tid in enumerate(gen_ids):
            if tid in stop_by_id:
                gen_ids = gen_ids[:idx + 1]
                stop_reason = stop_by_id[tid]
                break
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        expected = msgs[-1]["content"]
        match_icon = "✓" if expected.strip() in text.strip() else "✗"
        print(f"  [{match_icon}] Token {s.get('id', '?')}: {text[:100]}")
        print(f"       Expected: {expected}")
        print(f"       Stop: {stop_reason}, tokens: {len(gen_ids)}")
        print()


if __name__ == "__main__":
    main()
