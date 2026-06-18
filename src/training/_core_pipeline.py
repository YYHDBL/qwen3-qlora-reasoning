"""
Qwen3 SFT 数据处理核心流程（用你 data/processed/ 里的真实数据走一遍）

用法:
  cd qwen3-qlora-reasoning
  python src/training/_core_pipeline.py

你会看到每条数据: 原始 JSON → messages 格式 → chat template 渲染文本 → token_ids + mask → labels
"""

import json
import os
import sys

# 取项目根目录（脚本在 src/training/，需要往上退两层）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 把 src/training 目录加到 sys.path，让 chat_template 等模块可导入
sys.path.insert(0, os.path.join(ROOT, "src", "training"))
DATA_PATH = os.path.join(ROOT, "data", "instruction", "stage1", "train.jsonl")

# ─────────────────────────────────────────────
# 第 0 步: 从 No Robots JSONL 中拿一条真实数据
# ─────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.loads(f.readline())      # 第一条

print("=" * 60)
print("0. 原始 No Robots JSONL 中的一行 (id + messages):")
# 只打印关键字段，messages 太长只截取开头
clean = {k: raw[k] for k in ("id", "category", "messages")}
print(json.dumps(clean, indent=2, ensure_ascii=False)[:800])

# ─────────────────────────────────────────────
# 第 1 步: No Robots 数据已经是 messages 格式，直接用
# ─────────────────────────────────────────────
# 就取 messages 列表，与 chat template 输入格式一致
messages = raw["messages"]

# ─────────────────────────────────────────────
# 第 2 步: 加载 tokenizer + 注入训练 chat template
# ─────────────────────────────────────────────
from transformers import AutoTokenizer
from chat_template import configure_training_chat_template

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
configure_training_chat_template(tokenizer)

# ─────────────────────────────────────────────
# 第 3 步: 用训练模板渲染 → 看文本长啥样
# ─────────────────────────────────────────────
rendered_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,      # 只看文本，不转 ID
)

print("\n" + "=" * 60)
print("2. 训练 chat template 渲染后的文本:")
print(rendered_text)

# ─────────────────────────────────────────────
# 第 4 步: 渲染 + tokenize + mask 一步到位
# ─────────────────────────────────────────────
rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
    add_generation_prompt=False,
    # ↑ 训练时不需要生成 prompt，需要完整对话
)

input_ids = rendered["input_ids"]
# 兼容 transformers 不同版本：旧版用 assistant_tokens_mask，新版用 assistant_masks
mask = rendered.get("assistant_tokens_mask") or rendered.get("assistant_masks")
if mask is None:
    raise KeyError("chat template did not return assistant_tokens_mask / assistant_masks")
# 展平 batch 维度（batch=1 时可能返回嵌套列表）
if mask and isinstance(mask[0], list):
    mask = mask[0]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

print("\n" + "=" * 60)
print("3. tokenize 后的 input_ids 和 mask:")
print(f"   总 token 数: {len(input_ids)}")
print(f"   assistant token 数 (mask=1): {sum(mask)}")
print(f"   user/system token 数 (mask=0): {len(input_ids) - sum(mask)}")
print()

# 把 token 分段标出来，看清分界线
user_end = mask.index(1) if 1 in mask else len(input_ids)
print(f"   [user 部分] input_ids[:{user_end}] = {input_ids[:user_end]}")
print(f"   [assistant 部分] input_ids[{user_end}:] = {input_ids[user_end:]}")
print()
print(f"   mask  = {mask}")

# ─────────────────────────────────────────────
# 第 5 步: labels 构建
# ─────────────────────────────────────────────
labels = [
    token_id if assistant else -100
    for token_id, assistant in zip(input_ids, mask, strict=True)
]

print("\n" + "=" * 60)
print("4. labels（-100 = 不参与 loss）:")
print(f"   labels = {labels}")

assistant_start = labels.index(next(v for v in labels if v != -100))
print(f"   第一个非 -100 的位置: {assistant_start}")
print(f"   非 -100 的数量: {sum(1 for v in labels if v != -100)}")

# ─────────────────────────────────────────────
# 第 6 步: 解码验证
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 解码验证:")

decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
print(f"   完整序列解码: {repr(decoded_full[:200])}")

# 只保留 supervised 部分（assistant 回复）
supervised_ids = [t for t, l in zip(input_ids, labels) if l != -100]
decoded_sup = tokenizer.decode(supervised_ids, skip_special_tokens=False)
print(f"   仅 assistant 部分: {repr(decoded_sup[:200])}")

# ─────────────────────────────────────────────
# 第 7 步: 评估/推理用的生成 prompt
# ─────────────────────────────────────────────
# 推理时保留历史，只去掉最后一条 gold assistant 回复，
# add_generation_prompt=True 会在末尾加 "assistant\n" 等待模型补全
from chat_template import render_generation_prompt
if messages and messages[-1]["role"] == "assistant":
    gen_messages = messages[:-1]
else:
    gen_messages = messages
gen_prompt = render_generation_prompt(tokenizer, gen_messages)

print("\n" + "=" * 60)
print("6. 评估/推理时用的生成 prompt（保留历史，去掉最后答案）:")
print(gen_prompt)
print("  ↑ 以 <|im_start|>assistant\\n 结尾，模型从这里开始续写")

print("\n" + "=" * 60)
print("流程总结:")
print("  JSONL row → messages → 训练模板渲染 → tokenize + mask")
print("  → labels(assistant=原ID, 其他=-100) → 丢进模型算 loss")
