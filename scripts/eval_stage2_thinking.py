#!/usr/bin/env python3
"""Stage 2 Thinking Protocol Warmup - comprehensive evaluation.

Covers:
1. Thinking protocol eval (think-tag, stop, final-answer)
2. Strict regression eval (Stage 1.5 abilities preserved)
3. Open regression eval (No Robots abilities preserved)
4. protocol_test.jsonl as independent test set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Open prompts for regression ─────────────────────────────────────────────
OPEN_PROMPTS = [
    "Explain the difference between LoRA and full fine-tuning in 2-3 sentences.",
    "Summarize the following paragraph: The rapid advancement of large language models has transformed natural language processing. These models, trained on vast corpora of text, can perform tasks ranging from translation to code generation. However, their deployment raises concerns about computational cost, data privacy, and potential misuse.",
    "Rewrite this sentence professionally: 'Hey can u pls send me the report by tomorrow thx.'",
    "Give three reasons why retrieval-augmented generation helps large language models.",
    "Write a short Python function that reverses a list without using built-in reverse methods.",
    "What is the capital of France? Explain why it became the capital.",
    "Describe the water cycle in 3-4 sentences.",
    "Compare cats and dogs as pets. List 2 pros and 2 cons for each.",
    "Explain what a neural network is to a 10-year-old.",
    "If you could only recommend one book to someone learning programming, what would it be and why?",
    "What are the three main branches of the U.S. government and what does each do?",
    "Write a short haiku about programming.",
    "Explain why the sky is blue in simple terms.",
    "What is the difference between SQL and NoSQL databases? Give one use case for each.",
    "Describe a simple algorithm for sorting a list of numbers.",
    "What are microservices? Give one advantage and one disadvantage.",
    "Explain the concept of recursion with a simple example.",
    "What is the purpose of version control in software development?",
    "Name three renewable energy sources and briefly describe how each works.",
    "What does 'open source' mean in the context of software? Give an example.",
]


# ── Model loading ───────────────────────────────────────────────────────────
def load_model(config_path: str, adapter_path: str, for_training: bool = False):
    import torch
    from src.common.config import load_yaml_config
    from src.training.model_loader import load_bf16_model, load_lora_model, load_tokenizer
    from src.training.chat_template import resolve_stop_token_ids

    config = load_yaml_config(Path(config_path))
    tokenizer = load_tokenizer(config, for_training=for_training)
    model = load_lora_model(config, adapter_path, is_trainable=False).eval()
    model.config.use_cache = True
    stop_ids = resolve_stop_token_ids(tokenizer)
    return model, tokenizer, stop_ids


def generate(model, tokenizer, stop_ids, messages: list, max_new_tokens: int = 256,
             enable_thinking: bool | None = None) -> str:
    import torch
    from src.training.chat_template import render_generation_prompt

    full_messages = list(messages)
    prompt_text = render_generation_prompt(tokenizer, full_messages, enable_thinking=enable_thinking)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, stop_ids.get("im_end"), stop_ids.get("endoftext")],
        )
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()


# ── Protocol evaluation ─────────────────────────────────────────────────────
def evaluate_protocol_test(model, tokenizer, stop_ids, protocol_path: str, max_new_tokens: int = 256) -> dict:
    """Evaluate on protocol_test.jsonl."""
    records = []
    with open(protocol_path) as f:
        for line in f:
            r = json.loads(line.strip())
            records.append(r)

    results = []
    failures: dict[str, list[dict]] = defaultdict(list)
    stats = {
        "total": len(records),
        "think_requested": 0,
        "no_think_requested": 0,
        "think_tag_success": 0,
        "think_tag_failure": 0,
        "final_answer_parse_success": 0,
        "final_answer_parse_failure": 0,
        "stop_success": 0,
        "stop_failure": 0,
        "no_think_spurious": 0,
        "no_think_correct": 0,
        "mean_generated_tokens": 0.0,
        "overlong_thinking_rate": 0.0,
        "total_generated_tokens": 0,
    }

    iterator = tqdm(records, desc="  Protocol test", unit="sample") if tqdm else records
    for rec in iterator:
        messages = [m for m in rec["messages"] if m["role"] != "assistant"]
        metadata = rec.get("metadata", {})
        wants_think = metadata.get("thinking", False)

        # 关键修复：thinking prompt 开启 enable_thinking，no-think prompt 关闭
        think_flag = True if wants_think else False
        generated = generate(model, tokenizer, stop_ids, messages, max_new_tokens,
                             enable_thinking=think_flag)

        # Count tokens
        token_count = len(tokenizer.encode(generated))
        stats["total_generated_tokens"] += token_count

        has_think = "<think>" in generated
        has_close_think = "</think>" in generated
        has_both_tags = has_think and has_close_think

        # Determine stop success (did not hit max_new_tokens)
        hit_max = token_count >= max_new_tokens

        # Parse final answer (text after </think>)
        final_answer = ""
        think_content = ""
        if has_close_think:
            parts = generated.split("</think>", 1)
            if len(parts) >= 2:
                final_answer = parts[1].strip()
            think_content_split = generated.split("<think>", 1)
            if len(think_content_split) >= 2:
                before_close = think_content_split[1].split("</think>", 1)
                think_content = before_close[0].strip()
        else:
            final_answer = generated.strip()

        # Scoring
        record_result = {
            "id": rec["id"],
            "category": rec.get("category", ""),
            "wants_think": wants_think,
            "generated": generated,
            "has_think": has_think,
            "has_close_think": has_close_think,
            "has_both_tags": has_both_tags,
            "final_answer": final_answer,
            "token_count": token_count,
            "hit_max": hit_max,
        }

        if wants_think:
            stats["think_requested"] += 1
            if has_both_tags:
                stats["think_tag_success"] += 1
                if final_answer:
                    stats["final_answer_parse_success"] += 1
                else:
                    stats["final_answer_parse_failure"] += 1
                    failures["no_final_answer_after_think"].append(record_result)
            else:
                stats["think_tag_failure"] += 1
                if not has_think:
                    failures["missing_think_tag"].append(record_result)
                elif not has_close_think:
                    failures["missing_close_think"].append(record_result)
            if hit_max:
                stats["stop_failure"] += 1
                failures["stop_failure_thinking"].append(record_result)
            else:
                stats["stop_success"] += 1
            if len(think_content) > max_new_tokens * 0.7:
                stats["overlong_thinking_rate"] += 1
        else:
            stats["no_think_requested"] += 1
            if has_think:
                stats["no_think_spurious"] += 1
                failures["spurious_think"].append(record_result)
            else:
                stats["no_think_correct"] += 1
            if hit_max:
                stats["stop_failure"] += 1
                failures["stop_failure_nothink"].append(record_result)
            else:
                stats["stop_success"] += 1

        results.append(record_result)

    # Compute rates
    if stats["think_requested"] > 0:
        stats["think_tag_success_rate"] = stats["think_tag_success"] / stats["think_requested"]
        stats["final_answer_parse_rate"] = stats["final_answer_parse_success"] / stats["think_requested"]
        stats["overlong_thinking_rate"] = stats["overlong_thinking_rate"] / stats["think_requested"]
    else:
        stats["think_tag_success_rate"] = 0.0
        stats["final_answer_parse_rate"] = 0.0
        stats["overlong_thinking_rate"] = 0.0
    if stats["no_think_requested"] > 0:
        stats["no_think_when_not_requested_rate"] = stats["no_think_correct"] / stats["no_think_requested"]
    else:
        stats["no_think_when_not_requested_rate"] = 1.0
    if stats["total"] > 0:
        stats["stop_success_rate"] = stats["stop_success"] / stats["total"]
        stats["mean_generated_tokens"] = stats["total_generated_tokens"] / stats["total"]

    return {"stats": stats, "failures": failures, "results": results}


# ── Strict regression evaluation ────────────────────────────────────────────
def evaluate_strict_regression(model, tokenizer, stop_ids, strict_path: str, max_new_tokens: int = 128) -> dict:
    """Evaluate strict-format abilities using protocol_test stage1_5_strict_replay samples."""
    records = []
    with open(strict_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("category") == "stage1_5_strict_replay":
                records.append(r)

    stats = {
        "total": len(records),
        "exact_match": 0,
        "format_correct": 0,
        "no_spurious_think": 0,
        "spurious_think": 0,
        "stop_success": 0,
        "stop_failure": 0,
    }
    failures: dict[str, list[dict]] = defaultdict(list)

    iterator = tqdm(records, desc="  Strict regression", unit="sample") if tqdm else records
    for rec in iterator:
        messages = [m for m in rec["messages"] if m["role"] != "assistant"]
        expected = rec["messages"][-1]["content"].strip()
        generated = generate(model, tokenizer, stop_ids, messages, max_new_tokens,
                             enable_thinking=False)

        # Check for spurious think
        has_think = "<think>" in generated
        if has_think:
            stats["spurious_think"] += 1
            failures["strict_spurious_think"].append({"id": rec["id"], "expected": expected, "generated": generated})
        else:
            stats["no_spurious_think"] += 1

        # Exact match
        generated_clean = generated.strip()
        if generated_clean == expected:
            stats["exact_match"] += 1

        # Format check (basic heuristics)
        if not has_think:
            stats["format_correct"] += 1

        # Stop check
        token_count = len(tokenizer.encode(generated))
        if token_count >= max_new_tokens:
            stats["stop_failure"] += 1
            failures["strict_stop_failure"].append({"id": rec["id"], "generated": generated, "tokens": token_count})
        else:
            stats["stop_success"] += 1

    if stats["total"] > 0:
        stats["exact_match_rate"] = stats["exact_match"] / stats["total"]
        stats["no_think_rate"] = stats["no_spurious_think"] / stats["total"]
        stats["stop_success_rate"] = stats["stop_success"] / stats["total"]
        stats["format_correct_rate"] = stats["format_correct"] / stats["total"]

    return {"stats": stats, "failures": failures}


# ── Open regression evaluation ──────────────────────────────────────────────
def evaluate_open_regression(model, tokenizer, stop_ids, num_prompts: int = 20, max_new_tokens: int = 256) -> dict:
    """Evaluate open-ended answering quality."""
    prompts = OPEN_PROMPTS[:num_prompts]
    results = []
    stats = {
        "total": len(prompts),
        "has_think": 0,
        "no_think": 0,
        "stop_success": 0,
        "stop_failure": 0,
        "too_short": 0,
        "mean_tokens": 0.0,
        "total_tokens": 0,
    }
    failures: dict[str, list[dict]] = defaultdict(list)

    iterator = tqdm(prompts, desc="  Open regression", unit="prompt") if tqdm else prompts
    for prompt_text in iterator:
        messages = [{"role": "user", "content": prompt_text}]
        generated = generate(model, tokenizer, stop_ids, messages, max_new_tokens,
                             enable_thinking=False)
        token_count = len(tokenizer.encode(generated))

        has_think = "<think>" in generated

        if has_think:
            stats["has_think"] += 1
        else:
            stats["no_think"] += 1

        if token_count >= max_new_tokens:
            stats["stop_failure"] += 1
            failures["open_stop_failure"].append({"prompt": prompt_text, "generated": generated[:200]})
        else:
            stats["stop_success"] += 1

        # Too short = less than 10 tokens for open-ended questions
        if token_count < 10:
            stats["too_short"] += 1
            failures["open_too_short"].append({"prompt": prompt_text, "generated": generated})

        stats["total_tokens"] += token_count
        results.append({"prompt": prompt_text, "generated": generated, "tokens": token_count, "has_think": has_think})

    if stats["total"] > 0:
        stats["stop_success_rate"] = stats["stop_success"] / stats["total"]
        stats["mean_tokens"] = stats["total_tokens"] / stats["total"]
        stats["think_rate"] = stats["has_think"] / stats["total"]

    return {"stats": stats, "failures": failures, "results": results}


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Thinking Warmup Evaluation")
    parser.add_argument("--config", type=Path, required=True, help="Path to stage2 config YAML")
    parser.add_argument("--adapter-path", type=Path, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--protocol-test", type=Path, default=Path("data/instruction/stage2_thinking/protocol_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-open-prompts", type=int, default=20)
    parser.add_argument("--skip-open", action="store_true")
    parser.add_argument("--skip-strict", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 2 Thinking Warmup - Evaluation Report")
    print("=" * 60)

    # Load model
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, stop_ids = load_model(str(args.config), str(args.adapter_path))
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    all_failures: dict[str, list[dict]] = {}

    # 1. Protocol Test Eval
    print(f"\n[2/5] Protocol Test Evaluation ({args.protocol_test})...")
    t0 = time.time()
    proto_result = evaluate_protocol_test(model, tokenizer, stop_ids, str(args.protocol_test), args.max_new_tokens)
    proto_stats = proto_result["stats"]
    all_failures.update(proto_result["failures"])
    print(f"  Completed in {time.time() - t0:.1f}s")
    print(f"  Total samples: {proto_stats['total']}")
    print(f"  Think requested: {proto_stats['think_requested']} / No-think: {proto_stats['no_think_requested']}")
    print(f"  Think-tag success: {proto_stats['think_tag_success']}/{proto_stats['think_requested']} = {proto_stats['think_tag_success_rate']:.1%}")
    print(f"  Final-answer parse: {proto_stats['final_answer_parse_success']}/{proto_stats['think_requested']} = {proto_stats['final_answer_parse_rate']:.1%}")
    print(f"  No-think correct: {proto_stats['no_think_correct']}/{proto_stats['no_think_requested']} = {proto_stats['no_think_when_not_requested_rate']:.1%}")
    print(f"  Stop success: {proto_stats['stop_success']}/{proto_stats['total']} = {proto_stats['stop_success_rate']:.1%}")
    print(f"  Mean generated tokens: {proto_stats['mean_generated_tokens']:.1f}")
    print(f"  Overlong thinking rate: {proto_stats['overlong_thinking_rate']:.1%}")

    # 2. Strict Regression
    print(f"\n[3/5] Strict Regression Evaluation...")
    t0 = time.time()
    strict_result = evaluate_strict_regression(model, tokenizer, stop_ids, str(args.protocol_test), 128)
    strict_stats = strict_result["stats"]
    all_failures.update(strict_result["failures"])
    print(f"  Completed in {time.time() - t0:.1f}s")
    print(f"  Total strict samples: {strict_stats['total']}")
    print(f"  Exact match: {strict_stats['exact_match']}/{strict_stats['total']} = {strict_stats['exact_match_rate']:.1%}")
    print(f"  No spurious think: {strict_stats['no_spurious_think']}/{strict_stats['total']} = {strict_stats['no_think_rate']:.1%}")
    print(f"  Stop success: {strict_stats['stop_success']}/{strict_stats['total']} = {strict_stats['stop_success_rate']:.1%}")

    # 3. Open Regression
    print(f"\n[4/5] Open Regression Evaluation ({args.num_open_prompts} prompts)...")
    t0 = time.time()
    open_result = evaluate_open_regression(model, tokenizer, stop_ids, args.num_open_prompts, args.max_new_tokens)
    open_stats = open_result["stats"]
    all_failures.update(open_result["failures"])
    print(f"  Completed in {time.time() - t0:.1f}s")
    print(f"  Has think: {open_stats['has_think']}/{open_stats['total']} = {open_stats['think_rate']:.1%}")
    print(f"  Stop success: {open_stats['stop_success']}/{open_stats['total']} = {open_stats['stop_success_rate']:.1%}")
    print(f"  Too short: {open_stats['too_short']}")
    print(f"  Mean tokens: {open_stats['mean_tokens']:.1f}")

    # 5. Failure summary
    print(f"\n[5/5] Failure Summary")
    print("-" * 40)
    failure_labels = {
        "missing_think_tag": "Think requested, no <think>",
        "missing_close_think": "Has <think>, no </think>",
        "no_final_answer_after_think": "</think> but no final answer",
        "spurious_think": "No-think requested, but <think> appeared",
        "stop_failure_thinking": "Stop failure (thinking sample)",
        "stop_failure_nothink": "Stop failure (no-think sample)",
        "strict_spurious_think": "Strict: spurious <think>",
        "strict_stop_failure": "Strict: stop failure",
        "open_stop_failure": "Open: stop failure",
        "open_too_short": "Open: too short",
    }
    total_failures = 0
    for key, label in failure_labels.items():
        count = len(all_failures.get(key, []))
        if count > 0:
            print(f"  {label}: {count}")
            total_failures += count
    if total_failures == 0:
        print("  No failures!")

    # Write detailed results
    report = {
        "config": str(args.config),
        "adapter_path": str(args.adapter_path),
        "protocol_test_eval": proto_stats,
        "strict_regression_eval": strict_stats,
        "open_regression_eval": open_stats,
        "failures": {k: v[:10] for k, v in all_failures.items()},  # limit to 10 per category
        "open_regression_details": [r for r in open_result["results"]],
    }
    report_path = args.output_dir / "stage2_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}")

    # Print open regression details
    if open_result.get("results"):
        print("\n--- Open Regression Samples ---")
        for r in open_result["results"]:
            has_think_mark = " [THINK]" if r["has_think"] else ""
            print(f"\n  Prompt: {r['prompt'][:80]}...")
            print(f"  Response{has_think_mark} ({r['tokens']} tokens): {r['generated'][:200]}")

    # Print failure samples (first 3 per category)
    for key, samples in all_failures.items():
        if samples:
            print(f"\n--- Failure: {failure_labels.get(key, key)} ({len(samples)} total) ---")
            for s in samples[:3]:
                if "generated" in s:
                    print(f"  ID: {s.get('id', s.get('prompt', '?'))}")
                    print(f"  Generated: {s['generated'][:200]}")
                if "expected" in s:
                    print(f"  Expected: {s['expected']}")


if __name__ == "__main__":
    main()
