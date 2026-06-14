"""Shared BF16 Base and LoRA loading for Stage 1 training and evaluation."""

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
    status(f"Loading tokenizer: {model['id']}")
    tokenizer = AutoTokenizer.from_pretrained(model["id"], **kwargs)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer PAD token is unset; record and resolve explicitly")
    if for_training:
        tokenizer.padding_side = "right"
        configure_training_chat_template(tokenizer)
    else:
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
        "dtype": torch.bfloat16,
        "attn_implementation": model_config["attn_implementation"],
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }
    if model_config.get("revision"):
        kwargs["revision"] = model_config["revision"]
    status(f"Loading BF16 model: {model_config['id']}")
    model = AutoModelForCausalLM.from_pretrained(model_config["id"], **kwargs)
    model.config.use_cache = False
    return model


def load_lora_model(
    config: Mapping[str, Any],
    adapter_path: str,
    is_trainable: bool,
) -> Any:
    from peft import PeftModel

    base = load_bf16_model(config)
    status(f"Loading LoRA adapter: {adapter_path}")
    return PeftModel.from_pretrained(base, adapter_path, is_trainable=is_trainable)


def trim_generated_ids(
    generated_ids: Sequence[int],
    stop_ids: Mapping[str, int],
) -> tuple[list[int], str]:
    """Trim batch padding after the first generated stop token."""
    stop_by_id = {token_id: name for name, token_id in stop_ids.items()}
    for index, token_id in enumerate(generated_ids):
        if token_id in stop_by_id:
            return list(generated_ids[: index + 1]), stop_by_id[token_id]
    return list(generated_ids), "length"


class ChatGenerator:
    """Deterministic Qwen3 chat generation for Base and LoRA adapters."""

    def __init__(
        self,
        config: Mapping[str, Any],
        adapter_path: str | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.config = config
        self.tokenizer = load_tokenizer(config, for_training=False)
        self.model = (
            load_lora_model(config, adapter_path, is_trainable=False)
            if adapter_path
            else load_bf16_model(config)
        ).eval()
        self.tokenizer_revision = (
            getattr(self.tokenizer, "init_kwargs", {}) or {}
        ).get("_commit_hash") or getattr(self.tokenizer, "_commit_hash", None)
        self.model_revision = getattr(
            getattr(self.model, "config", None), "_commit_hash", None
        )
        self.stop_ids = resolve_stop_token_ids(self.tokenizer)

    def generate(
        self,
        conversations: Sequence[Sequence[Mapping[str, str]]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        results: list[dict[str, Any]] = []
        for start in range(0, len(conversations), batch_size):
            batch = conversations[start : start + batch_size]
            prompts = [
                render_generation_prompt(self.tokenizer, messages) for messages in batch
            ]
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.config["training"]["max_length"]),
                add_special_tokens=False,
            )
            encoded = {
                key: value.to(self.model.device) for key, value in encoded.items()
            }
            input_width = encoded["input_ids"].shape[1]
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **encoded,
                    do_sample=bool(self.config["generation"].get("do_sample", False)),
                    max_new_tokens=int(self.config["generation"]["max_new_tokens"]),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=list(self.stop_ids.values()),
                )
            for output in outputs:
                generated_ids, stop_reason = trim_generated_ids(
                    output[input_width:].tolist(), self.stop_ids
                )
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
            status(
                f"Generated {min(start + len(batch), len(conversations))}/"
                f"{len(conversations)}"
            )
        return results
