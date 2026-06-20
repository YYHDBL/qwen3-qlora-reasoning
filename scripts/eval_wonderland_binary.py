#!/usr/bin/env python3
"""Stage 1.5 wonderland-like binary eval: check model outputs only 8-bit binary with no explanation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Synthetic 8-bit transformation prompts (NOT official Wonderland data)
BINARY_PROMPTS = [
    "Given examples: 10101010 -> 01010101, 11110000 -> 00001111. Apply the rule: 11001100 ->",
    "Transform this 8-bit string using the same rule as: 00001111 -> 11110000. Now transform: 10101010 ->",
    "Flip all bits: 00110011 ->",
    "Reverse the bits: 10110100 ->",
    "Rotate left by 1: 01111110 ->",
    "Shift right by 2 and pad left with 0: 11010001 ->",
    "XOR with 01010101: 10101010 ->",
    "AND with 11110000: 11001101 ->",
    "OR with 00001111: 10100000 ->",
    "Invert and add 1: 00000001 ->",
    "Swap first 4 and last 4 bits: 10101100 ->",
    "Output the 2's complement negation: 00001010 ->",
    "Given example: 01010101 -> 10101010. Apply: 11111111 ->",
    "Given example: 00111100 -> 11000011. Apply: 10011001 ->",
    "Given example: 00011000 -> 11100111. Apply: 01100110 ->",
    "Given example: 11100001 -> 00011110. Apply: 10000001 ->",
    "Given example: 01101101 -> 10010010. Apply: 11011011 ->",
    "Given example: 00010001 -> 11101110. Apply: 10111011 ->",
    "Given example: 11000110 -> 00111001. Apply: 01110111 ->",
    "Given example: 10111101 -> 01000010. Apply: 11101110 ->",
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


def check_binary_format(response: str) -> dict[str, Any]:
    stripped = response.strip()
    result = {
        "is_8bit_binary": bool(re.fullmatch(r"[01]{8}", stripped)),
        "contains_explanation": any(
            kw in response.lower()
            for kw in ["explanation", "because", "therefore", "the rule", "i think", "let me"]
        ),
        "contains_markdown": "```" in response or "**" in response,
        "has_extra_content": len(stripped) > 8 and bool(re.fullmatch(r"[01]{8}", stripped)),
        "raw_length": len(stripped),
    }
    result["format_pass"] = result["is_8bit_binary"] and not result["contains_explanation"] and not result["contains_markdown"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Wonderland-like binary format eval for Stage 1.5")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=20)
    args = parser.parse_args()

    import torch
    import time

    model, tokenizer, stop_ids = load_model(
        str(args.config), str(args.adapter_path) if args.adapter_path else None, False
    )

    prompts = BINARY_PROMPTS[: args.num_prompts]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    format_pass = 0

    for i, prompt_text in enumerate(prompts):
        history = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=list(stop_ids.values()),
            )

        response = tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        check = check_binary_format(response)
        if check["format_pass"]:
            format_pass += 1

        results.append({
            "index": i,
            "prompt": prompt_text,
            "response": response,
            **check,
        })

    summary = {
        "model": str(args.adapter_path) if args.adapter_path else "base",
        "num_prompts": len(results),
        "format_pass_count": format_pass,
        "format_pass_rate": format_pass / len(results) if results else 0,
    }

    with open(output_dir / "predictions.json", "w") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "report.md", "w") as f:
        f.write("# Wonderland-Like Binary Format Eval\n\n")
        f.write(f"- Model: {summary['model']}\n")
        f.write(f"- Prompts: {summary['num_prompts']}\n")
        f.write(f"- Format pass rate: {summary['format_pass_rate']:.2%}\n\n")
        for r in results:
            status = "PASS" if r["format_pass"] else "FAIL"
            f.write(f"## {r['index'] + 1} [{status}]\n\n")
            f.write(f"**Q:** {r['prompt']}\n\n")
            f.write(f"**A:** `{r['response']}`\n\n")
            issues = []
            if r["contains_explanation"]:
                issues.append("explanation")
            if r["contains_markdown"]:
                issues.append("markdown")
            if r["has_extra_content"]:
                issues.append("extra content after 8 bits")
            if issues:
                f.write(f"Issues: {', '.join(issues)}\n\n")
            f.write("---\n\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nFull report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
