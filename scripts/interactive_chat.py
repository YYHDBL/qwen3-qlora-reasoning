#!/usr/bin/env python3
"""交互式对话测试 Stage 2 思考能力。

命令:
  /think    — 开启思考模式 (enable_thinking=True)
  /nothink  — 关闭思考模式 (enable_thinking=False)
  /clear    — 清除对话历史
  /quit     — 退出
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "models/Qwen3-1.7B-Base"
ADAPTER = "outputs/stage2_thinking_warmup/formal/adapter"
MAX_NEW = 512

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False).eval()
stop_ids = [tokenizer.eos_token_id, 151645]  # eos + im_end
print("Ready.\n")

enable_thinking = True
history: list[dict] = []

print("Commands: /think  /nothink  /clear  /quit")
print(f"Thinking: {'ON' if enable_thinking else 'OFF'}\n")

while True:
    try:
        user_input = input("You> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_input:
        continue

    if user_input == "/quit":
        break
    elif user_input == "/think":
        enable_thinking = True
        print(f"[Thinking: ON]\n")
        continue
    elif user_input == "/nothink":
        enable_thinking = False
        print(f"[Thinking: OFF]\n")
        continue
    elif user_input == "/clear":
        history = []
        print("[History cleared]\n")
        continue

    history.append({"role": "user", "content": user_input})

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prompt = tokenizer.apply_chat_template(history, **kwargs)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids,
        )

    input_len = inputs.input_ids.shape[1]
    response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    history.append({"role": "assistant", "content": response})
    print(f"Bot> {response}\n")
