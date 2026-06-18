"""Stage 1 训练和评估共享的 BF16 Base 和 LoRA 加载。"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Mapping, Sequence

from .chat_template import (
    configure_training_chat_template,
    render_generation_prompt,
    resolve_stop_token_ids,
)


def status(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] [stage1] {message}", file=sys.stderr, flush=True)


def load_tokenizer(config: Mapping[str, Any], for_training: bool) -> Any:
    from transformers import AutoTokenizer

    model = config["model"]
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model.get("trust_remote_code", False))
    }
    revision = model.get("tokenizer_revision") or model.get("revision")
    if revision:
        kwargs["revision"] = revision
    cache_dir = model.get("cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    status(f"Loading tokenizer: {model['id']}")
    tokenizer = AutoTokenizer.from_pretrained(model["id"], **kwargs)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer PAD token is unset; record and resolve explicitly")
    if for_training:
        # 训练时 padding_side="right":
        #   - 序列右侧补 pad token，序列内容在左侧对齐
        #   - 自回归 loss 计算时，pad 位置对应的 labels=-100 自动跳过
        #   - 如果 left padding，labels 对齐会错位，loss 计算出错
        tokenizer.padding_side = "right"
        # 注入训练用聊天模板（含 {% generation %} 标记，支持 assistant-only loss）
        configure_training_chat_template(tokenizer)
    else:
        # 生成时 padding_side="left":
        #   - 序列左侧补 pad token，prompt 内容在右侧对齐
        #   - 自回归生成从右往左读取，左侧 padding 不会干扰生成内容
        #   - 如果 right padding，batch 中短序列的 prompt 被推到中间，模型会"看到"右侧的 pad 再开始生成
        tokenizer.padding_side = "left"
    return tokenizer


def load_bf16_model(config: Mapping[str, Any]) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BF16 Stage 1")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the active CUDA device does not support BF16")
    model_config = config["model"]
    kwargs: dict[str, Any] = {
        # BF16：计算和存储精度为 bfloat16，数值范围与 FP32 相同，不易溢出
        "dtype": torch.bfloat16,
        "attn_implementation": model_config["attn_implementation"],
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        # low_cpu_mem_usage=True:
        #   - 加载模型时只创建一个 tensor 副本，而非 HuggingFace 默认的"创建后复制"
        #   - 例如加载 7B 模型在 FP16 下约 14GB，默认方式需要 28GB CPU 内存（两次拷贝）
        #   - 开启后仅需 ~14GB，对于消费级 CPU（32-64GB RAM）至关重要
        #   - 缺点：无法直接从 PyTorch checkpoint 恢复（但这里仅用于加载预训练权重，无此需求）
        "low_cpu_mem_usage": True,
        # device_map="auto" 自动将模型分层放置到 GPU（单 GPU 则全部放 GPU）
        # 训练时 SFTTrainer 会接管设备管理，此处传入不影响训练流程
        "device_map": "auto",
    }
    if model_config.get("revision"):
        kwargs["revision"] = model_config["revision"]
    cache_dir = model_config.get("cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    status(f"Loading BF16 model: {model_config['id']}")
    model = AutoModelForCausalLM.from_pretrained(model_config["id"], **kwargs)
    # 训练时关闭 KV cache：每步前向都重新计算，不需要缓存历史 key/value
    # 否则反向传播时梯度图会包含缓存的中间状态，导致显存急剧膨胀
    model.config.use_cache = False
    return model


def load_lora_model(
    config: Mapping[str, Any],
    adapter_path: str,
    is_trainable: bool,
) -> Any:
    from peft import PeftModel

    # 第 1 步：加载 BF16 base 模型（与训练时完全相同的加载方式）
    base = load_bf16_model(config)
    status(f"Loading LoRA adapter: {adapter_path}")
    # 第 2 步：在 base 模型上挂载 LoRA adapter 权重
    #   - is_trainable=False: adapter 权重冻结，仅用于推理/评估
    #   - PEFT 内部会读取 adapter_config.json 确定 r、alpha、target_modules 等信息
    #   - 加载的 adapter 权重与训练时保存的一致（trainer.save_model 保存 adapter 目录）
    adapter_kwargs: dict[str, Any] = {}
    cache_dir = config["model"].get("cache_dir")
    if cache_dir:
        adapter_kwargs["cache_dir"] = cache_dir
    return PeftModel.from_pretrained(
        base, adapter_path, is_trainable=is_trainable, **adapter_kwargs
    )


def trim_generated_ids(
    generated_ids: Sequence[int],
    stop_ids: Mapping[str, int],
) -> tuple[list[int], str]:
    """裁剪掉第一个停止 token 之后的 batch padding。

    返回 ``(trimmed_ids, stop_reason)``，其中 ``stop_reason`` 是触发的
    停止 token 名称，若未触发任何停止 token 则为 ``"length"``。
    """
    # 构建 token_id -> stop_name 的逆向映射，用于确定触发了哪个停止 token
    stop_by_id = {token_id: name for name, token_id in stop_ids.items()}
    # 遍历生成的每个 token，找到第一个匹配的停止 token
    for index, token_id in enumerate(generated_ids):
        if token_id in stop_by_id:
            # 保留停止 token 本身（含 im_end 可帮助下游解析），截断后续内容
            return list(generated_ids[: index + 1]), stop_by_id[token_id]
    # 未找到任何停止 token：说明模型在 max_new_tokens 耗尽前未触发停止
    # 返回完整的生成序列，标记为 "length"（达到生成长度上限）
    return list(generated_ids), "length"


class ChatGenerator:
    """Deterministic Qwen3 chat generation for Base and LoRA adapters."""

    def __init__(
        self,
        config: Mapping[str, Any],
        adapter_path: str | None = None,
    ) -> None:
        import torch

        # 保存 torch 模块引用，供 generate 方法内使用
        self.torch = torch
        # tqdm: 可选依赖，若未安装则不显示进度条
        self.tqdm = None
        try:
            from tqdm import tqdm as _tqdm
            self.tqdm = _tqdm
        except ImportError:
            pass
        self.config = config
        # tokenizer: for_training=False => padding_side="left"（生成用左侧 padding）
        self.tokenizer = load_tokenizer(config, for_training=False)
        # model: 若有 adapter_path 则加载 LoRA adapter，否则直接用 base 模型
        # 统一设为 eval() 模式：关闭 dropout、关闭 BN 更新
        self.model = (
            load_lora_model(config, adapter_path, is_trainable=False)
            if adapter_path
            else load_bf16_model(config)
        ).eval()
        self.model.config.use_cache = True
        # 记录 tokenizer 的 commit hash，用于环境追踪（非关键逻辑，仅用于审计）
        self.tokenizer_revision = (
            getattr(self.tokenizer, "init_kwargs", {}) or {}
        ).get("_commit_hash") or getattr(self.tokenizer, "_commit_hash", None)
        # 记录模型的 commit hash
        self.model_revision = getattr(
            getattr(self.model, "config", None), "_commit_hash", None
        )
        # 解析停止 token ID 映射: {"im_end": 151645, "endoftext": 151643}
        self.stop_ids = resolve_stop_token_ids(self.tokenizer)

    def generate(
        self,
        conversations: Sequence[Sequence[Mapping[str, str]]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        results: list[dict[str, Any]] = []
        total = len(conversations)
        # 计算总 batch 数（向上取整）
        num_batches = (total + batch_size - 1) // batch_size
        # 如果 tqdm 可用，初始化进度条
        pbar = self.tqdm(
            total=total, desc="Generating", unit="sample"
        ) if self.tqdm else None
        # 逐 batch 处理
        for start in range(0, total, batch_size):
            # ── 步骤 1: 取当前 batch ──
            batch = conversations[start : start + batch_size]
            # ── 步骤 2: 将对话列表渲染为 prompt 文本 ──
            # render_generation_prompt: 用训练 chat template 渲染，add_generation_prompt=True
            #   => 输出末尾附加 "<|im_start|>assistant\n"，等待模型补全
            prompts = [
                render_generation_prompt(self.tokenizer, messages) for messages in batch
            ]
            # ── 步骤 3: tokenize prompt ──
            #   - padding=True: 将 batch 内不同长度的 prompt 对齐
            #   - truncation=True+max_length: 超出训练 max_length 的 prompt 截断
            #   - add_special_tokens=False: chat template 已包含特殊 token，不重复加
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.config["training"]["max_length"]),
                add_special_tokens=False,
            )
            # ── 步骤 4: 将输入移到模型所在的 GPU 设备 ──
            encoded = {
                key: value.to(self.model.device) for key, value in encoded.items()
            }
            # 记录 prompt 的 token 数，后面用来从 output 中切掉 prompt 部分
            input_width = encoded["input_ids"].shape[1]
            # ── 步骤 5: 自回归生成 ──
            #   - inference_mode: 关闭梯度计算，减少显存占用
            #   - do_sample: 从 YAML 读取，默认 False（贪心解码，确定性输出）
            #   - max_new_tokens: 限制生成长度
            #   - eos_token_id: 停止条件列表（im_end + eos），命中任一即停止当前序列
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **encoded,
                    do_sample=bool(self.config["generation"].get("do_sample", False)),
                    max_new_tokens=int(self.config["generation"]["max_new_tokens"]),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=list(self.stop_ids.values()),
                )
            # ── 步骤 6: 后处理 ──
            for output in outputs:
                # 切掉 prompt 部分，只保留新生成 token；trim 到第一个停止 token
                generated_ids, stop_reason = trim_generated_ids(
                    output[input_width:].tolist(), self.stop_ids
                )
                # 解码为文本，记录统计信息
                results.append(
                    {
                        "text": self.tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "generated_tokens": len(generated_ids),
                        "stop_reason": stop_reason,
                    }
                )
            # 更新进度条
            if pbar is not None:
                pbar.update(len(batch))
        # 关闭进度条
        if pbar is not None:
            pbar.close()
        return results
