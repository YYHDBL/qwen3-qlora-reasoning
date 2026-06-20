#!/usr/bin/env python3
"""Stage 1.5 open-prompt regression eval: check model does not degenerate to one-word answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def load_model(config_path: str, adapter_path: str | None, for_training: bool):
    import torch
    from src.common.config import load_yaml_config
    from src.training.model_loader import load_bf16_model, load_lora_model, load_tokenizer

    config = load_yaml_config(Path(config_path))
    tokenizer = load_tokenizer(config, for_training=False)

    if adapter_path:
        model = load_lora_model(config, adapter_path, is_trainable=False).eval()
    else:
        model = load_bf16_model(config).eval()
    model.config.use_cache = True

    from src.training.chat_template import resolve_stop_token_ids
    stop_ids = resolve_stop_token_ids(tokenizer)
    return model, tokenizer, stop_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-prompt regression eval for Stage 1.5")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-prompts", type=int, default=10)
    args = parser.parse_args()

    import torch
    import time

    model, tokenizer, stop_ids = load_model(
        str(args.config), str(args.adapter_path) if args.adapter_path else None, False
    )

    prompts = OPEN_PROMPTS[: args.num_prompts]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total_time = 0.0

    for i, prompt_text in enumerate(prompts):
        history = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = args.temperature > 0
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "do_sample": do_sample,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": list(stop_ids.values()),
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = 0.9

        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**gen_kwargs)
        elapsed = time.monotonic() - start
        total_time += elapsed

        response = tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        word_count = len(response.split())
        char_count = len(response)

        # heuristic quality flags
        flags = []
        if word_count <= 3:
            flags.append("TOO_SHORT")
        if char_count >= args.max_new_tokens * 3:
            flags.append("POSSIBLE_TRUNCATION")
        if response.lower().strip() in {"i don't know", "i do not know", "idk", "n/a"}:
            flags.append("REFUSAL")

        results.append({
            "index": i,
            "prompt": prompt_text,
            "response": response,
            "word_count": word_count,
            "char_count": char_count,
            "elapsed_sec": elapsed,
            "flags": flags,
        })

    # summary
    word_counts = [r["word_count"] for r in results]
    too_short = sum(1 for r in results if "TOO_SHORT" in r["flags"])

    summary = {
        "model": str(args.adapter_path) if args.adapter_path else "base",
        "num_prompts": len(results),
        "total_elapsed_sec": total_time,
        "word_count_stats": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": sum(word_counts) / len(word_counts),
        },
        "too_short_count": too_short,
        "too_short_rate": too_short / len(results) if results else 0,
    }

    # write artifacts
    with open(output_dir / "predictions.json", "w") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # human-readable report
    with open(output_dir / "report.md", "w") as f:
        f.write("# Open-Prompt Regression Eval\n\n")
        f.write(f"- Model: {summary['model']}\n")
        f.write(f"- Prompts: {summary['num_prompts']}\n")
        f.write(f"- Mean words: {summary['word_count_stats']['mean']:.1f}\n")
        f.write(f"- Too short (<4 words): {summary['too_short_count']}/{summary['num_prompts']}\n")
        f.write(f"- Total time: {total_time:.1f}s\n\n")
        for r in results:
            flags_str = f" [{', '.join(r['flags'])}]" if r["flags"] else ""
            f.write(f"## Prompt {r['index'] + 1}{flags_str}\n\n")
            f.write(f"**Q:** {r['prompt']}\n\n")
            f.write(f"**A:** {r['response']}\n\n")
            f.write(f"*({r['word_count']} words, {r['char_count']} chars, {r['elapsed_sec']:.1f}s)*\n\n")
            f.write("---\n\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nFull report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
