"""Interactive chat demo for base / LoRA-adapted Qwen3 models.

Usage::

    # Base model (4B from HF hub)
    python -m src.models.chat_demo --config configs/stage1_no_robots.yaml

    # Local 1.7B model
    python -m src.models.chat_demo --config configs/stage1_no_robots_qwen3_1_7b_local.yaml

    # With LoRA adapter
    python -m src.models.chat_demo --config configs/stage1_no_robots.yaml \\
        --adapter-path outputs/stage1_no_robots/smoke/adapter
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TextStreamer

from src.training.model_loader import (
    load_bf16_model,
    load_lora_model,
    load_tokenizer,
)
from src.common.config import load_yaml_config
from src.training.chat_template import resolve_stop_token_ids


AUTO_PROMPTS = [
    ("你擅长什么？——用一句话回答", "general"),
    ("请用Python写一个计算斐波那契数列的函数", "code"),
    ("解释什么是机器学习，用两句话", "knowledge"),
    ("比较一下猫和狗作为宠物的优缺点", "compare"),
    ("请推荐三本值得读的书并简单说明理由", "recommend"),
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InteractiveChat:
    def __init__(
        self,
        config: dict[str, Any],
        adapter_path: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        open_thinking: bool = False,
    ) -> None:
        self.config = config
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.open_thinking = open_thinking
        self.tokenizer = load_tokenizer(config, for_training=False)
        self.stop_ids = resolve_stop_token_ids(self.tokenizer)
        if adapter_path:
            self.model = load_lora_model(
                config, adapter_path, is_trainable=False
            ).eval()
        else:
            self.model = load_bf16_model(config).eval()
        self.model.config.use_cache = True

    def chat(
        self,
        history: list[dict[str, str]],
        user_input: str,
    ) -> tuple[str, float]:
        history.append({"role": "user", "content": user_input})
        prompt = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.open_thinking,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.config["training"]["max_length"]),
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.stop_ids.values()),
                streamer=streamer,
            )
        elapsed = time.monotonic() - start

        new_tokens = output.shape[1] - prompt_len
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else float("inf")

        response = self.tokenizer.decode(
            output[0][prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        history.append({"role": "assistant", "content": response})
        return response, tokens_per_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Qwen3 chat demo (base or LoRA adapter)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_no_robots.yaml",
        help="Path to Stage 1 YAML config.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter directory.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per response (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9).",
    )
    parser.add_argument(
        "--open-thinking",
        action="store_true",
        help="Enable <think> reasoning block in generation.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=4,
        help="Number of past conversation turns to keep (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    model_id = config["model"]["id"]
    mode = "LoRA" if args.adapter_path else "Base"

    print(f"  Model: {model_id}")
    print(f"  Mode:  {mode}")
    if args.adapter_path:
        print(f"  Adapter: {args.adapter_path}")
    print(f"  Temp={args.temperature}  top_p={args.top_p}  max_tokens={args.max_new_tokens}")
    print(f"  Thinking={'on' if args.open_thinking else 'off'}")
    print()

    chat = InteractiveChat(
        config,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        open_thinking=args.open_thinking,
    )

    print("Choose mode:")
    print("  [0] Auto test (preset prompts)")
    print("  [1] Interactive chat")
    try:
        mode_choice = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if mode_choice == "0":
        _seed_everything(args.seed)
        history: list[dict[str, str]] = []
        for prompt, category in AUTO_PROMPTS:
            print(f"\n{'='*50}")
            print(f"[{category}] 💬: {prompt}")
            print(f"{'='*50}")
            print("🧠: ", end="", flush=True)
            _, tps = chat.chat(history, prompt)
            print(f"\n--- {tps:.1f} tokens/s ---")
            history = history[-args.history * 2 :]
    else:
        history = []
        print("\nType /clear to reset history, /exit to quit.\n")
        while True:
            try:
                user_input = input("💬: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user_input:
                continue
            if user_input == "/exit":
                break
            if user_input == "/clear":
                history = []
                print("History cleared.\n")
                continue
            print("🧠: ", end="", flush=True)
            _, tps = chat.chat(history, user_input)
            print(f"\n── {tps:.1f} tokens/s ──\n")
            history = history[-args.history * 2 :]
    print("Done.")


if __name__ == "__main__":
    main()
