"""Model inspection and diagnostics utility.

Usage::

    python -m src.models.inspect_model --config configs/stage1_no_robots.yaml
    python -m src.models.inspect_model --config configs/stage1_no_robots.yaml \\
        --adapter-path outputs/stage1_no_robots/smoke/adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _section(title: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}\n  {title}\n{sep}")


def _kv(key: str, value: Any) -> None:
    print(f"  {key:35s} {value}")


def _load_config(config_path: str) -> dict[str, Any]:
    from src.common.config import load_yaml_config

    config = load_yaml_config(Path(config_path))
    return dict(config)


def inspect_tokenizer(config: dict[str, Any]) -> Any:
    """Load tokenizer and print diagnostic information."""
    from src.training.model_loader import load_tokenizer

    _section("Tokenizer")
    tokenizer = load_tokenizer(config, for_training=False)

    _kv("tokenizer class", type(tokenizer).__name__)
    _kv("model path / id", config["model"]["id"])
    _kv("vocab_size", tokenizer.vocab_size)
    _kv(
        "vocab_size (len)",
        len(tokenizer),
    )
    _kv("pad_token", f"{tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    _kv("eos_token", f"{tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    _kv("bos_token", f"{tokenizer.bos_token!r}  (id={tokenizer.bos_token_id})")
    _kv("unk_token", f"{tokenizer.unk_token!r}  (id={tokenizer.unk_token_id})")

    chat_template = tokenizer.get_chat_template()
    has_template = bool(chat_template)
    _kv("chat_template exists", has_template)
    if has_template and chat_template is not None:
        _kv("chat_template length", f"{len(chat_template)} chars")
        lines = chat_template.strip().split("\n")
        _kv("chat_template lines", len(lines))
        _kv(
            "chat_template preview",
            lines[0][:60] + "..." if lines[0] else "(empty)",
        )

    extra_ids = []
    for name in ("im_start", "im_end"):
        tok = f"<|{name}|>"
        tid = tokenizer.convert_tokens_to_ids(tok)
        unk = tokenizer.unk_token_id
        extra_ids.append(f"{tok} => {tid} {'(UNK!)' if tid == unk else ''}")
    _kv("special tokens", ", ".join(extra_ids))

    system_tokens = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
    _kv(
        "im_start system token ids",
        system_tokens if len(system_tokens) <= 8 else f"{system_tokens[:8]}...",
    )
    return tokenizer


def inspect_model_architecture(config: dict[str, Any]) -> dict[str, Any]:
    """Load model *config only* (no weights) and print architecture info."""
    from transformers import AutoConfig

    _section("Model Architecture (config only, no weights loaded)")

    model_id = config["model"]["id"]
    cache_dir = config["model"].get("cache_dir")
    kwargs: dict[str, Any] = {"trust_remote_code": bool(config["model"].get("trust_remote_code", False))}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    model_cfg = AutoConfig.from_pretrained(model_id, **kwargs)

    _kv("model class", model_cfg.__class__.__name__)
    _kv("model path / id", model_id)
    _kv("architectures", str(model_cfg.architectures) if hasattr(model_cfg, "architectures") else "N/A")
    _kv("hidden_size", getattr(model_cfg, "hidden_size", "N/A"))
    _kv("num_hidden_layers", getattr(model_cfg, "num_hidden_layers", "N/A"))
    _kv("num_attention_heads", getattr(model_cfg, "num_attention_heads", "N/A"))
    _kv("num_key_value_heads", getattr(model_cfg, "num_key_value_heads", "N/A"))
    _kv("intermediate_size", getattr(model_cfg, "intermediate_size", "N/A"))
    _kv("max_position_embeddings", getattr(model_cfg, "max_position_embeddings", "N/A"))
    _kv("vocab_size", getattr(model_cfg, "vocab_size", "N/A"))
    _kv("torch dtype (config)", str(getattr(model_cfg, "torch_dtype", config["model"].get("dtype", "N/A"))))
    _kv("tie_word_embeddings", getattr(model_cfg, "tie_word_embeddings", "N/A"))
    rope = getattr(model_cfg, "rope_scaling", None)
    _kv("rope_scaling", str(rope) if rope else "none (default)")
    return dict(model_cfg.to_dict())


def _format_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def inspect_model_params(config: dict[str, Any]) -> None:
    """Load full model weights and count parameters / Linear module names."""
    from src.training.model_loader import load_bf16_model

    _section("Model Parameters (loading weights)")

    model = load_bf16_model(config)
    total = sum(p.numel() for p in model.parameters())
    _kv("total parameters", f"{total:,}  ({_format_count(total)})")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _kv("trainable (base, no LoRA)", f"{total_trainable:,}  ({_format_count(total_trainable)})")

    _section("Linear Module Names")
    linear_modules: dict[str, int] = {}
    for name, module in model.named_modules():
        if "Linear" in type(module).__name__:
            canonical = name.split(".")[-1]
            linear_modules[canonical] = linear_modules.get(canonical, 0) + 1
    for mod_name in sorted(linear_modules):
        _kv(f"  {mod_name}", f"x{linear_modules[mod_name]}")
    _kv("unique Linear names", list(sorted(linear_modules)))
    _kv("total Linear modules", sum(linear_modules.values()))

    # LoRA target module matching
    lora_target = config["lora"]["target_modules"]
    _section("LoRA Target Module Matching")
    _kv("configured target_modules", lora_target)
    if lora_target == "all-linear":
        _kv("resolved targets (all Linear)", list(sorted(linear_modules)))
        _kv("would match", f"{sum(linear_modules.values())} modules")
    elif isinstance(lora_target, list):
        matched = [m for m in lora_target if m in linear_modules]
        missing = [m for m in lora_target if m not in linear_modules]
        _kv("matched", str(matched) if matched else "(none)")
        _kv("would inject", f"{sum(linear_modules.get(m, 0) for m in matched)} modules")
        if missing:
            _kv("UNMATCHED", str(missing))
    _kv("", "")
    _kv("LoRA config", json.dumps(config["lora"], indent=2).replace("\n", "\n" + " " * 37))

    del model


def inspect_adapter(config: dict[str, Any], adapter_path: str) -> None:
    """Load a LoRA adapter, verify it, and run a smoke forward pass."""
    import torch
    from src.training.model_loader import load_bf16_model, load_tokenizer
    from peft import PeftModel

    _section("Adapter Inspection")

    adapter_dir = Path(adapter_path)
    adapter_config_file = adapter_dir / "adapter_config.json"
    _kv("adapter-path", str(adapter_dir))
    _kv("adapter_config.json exists", adapter_config_file.is_file())
    if not adapter_config_file.is_file():
        print("  ERROR: adapter_config.json not found — cannot load adapter")
        return
    adapter_cfg = json.loads(adapter_config_file.read_text(encoding="utf-8"))
    _kv("adapter type", adapter_cfg.get("peft_type", "unknown"))
    _kv("r (rank)", adapter_cfg.get("r", "N/A"))
    _kv("lora_alpha", adapter_cfg.get("lora_alpha", "N/A"))
    _kv("lora_dropout", adapter_cfg.get("lora_dropout", "N/A"))
    _kv("target_modules", str(adapter_cfg.get("target_modules", "N/A")))

    _kv("", "")
    _kv("Loading base model + adapter ...", "")
    base = load_bf16_model(config)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)

    lora_modules = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_modules += 1
            if lora_modules <= 5:
                _kv(f"  {name}", type(module).__name__)
    if lora_modules > 5:
        _kv(f"  ... and {lora_modules - 5} more", f"(total {lora_modules} LoRA layers)")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(
        p.numel() for n, p in model.named_parameters() if "lora_" in n
    )
    base_total = sum(
        p.numel() for p in model.base_model.parameters()
    )
    _kv("base model params", f"{base_total:,}  ({_format_count(base_total)})")
    _kv("LoRA adapter params", f"{adapter_params:,}  ({_format_count(adapter_params)})")
    _kv("total params (base+LoRA)", f"{total:,}  ({_format_count(total)})")
    _kv("trainable params (LoRA)", f"{trainable:,}  ({_format_count(trainable)})")
    ratio = trainable / total * 100 if total > 0 else 0
    _kv("trainable ratio", f"{ratio:.2f}%")
    _kv("note", "(is_trainable=False: adapter loaded frozen for inference/smoke test)")
    _kv("adapter load status", "SUCCESS")

    # Smoke forward pass
    _kv("", "")
    _kv("Smoke forward pass ...", "")
    tokenizer = load_tokenizer(config, for_training=False)
    prompt = render_generation_prompt(
        tokenizer,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ],
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    _kv("prompt preview", prompt[:80].replace("\n", "\\n") + "...")
    _kv("generated text", generated[-120:].replace("\n", "\\n"))
    _kv("smoke test", "PASSED")

    del model, base


def render_generation_prompt(
    tokenizer: Any, messages: list[dict[str, str]]
) -> str:
    """Render a conversation as a prompt string."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect model, tokenizer, and optional LoRA adapter."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_no_robots.yaml",
        help="Path to Stage 1 YAML config (default: configs/stage1_no_robots.yaml)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional path to LoRA adapter directory for inspection + smoke test",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip loading full model weights (faster, but no param count or module mapping)",
    )
    args = parser.parse_args()

    print(f"Inspect Model Report")
    print(f"  config: {args.config}")
    if args.adapter_path:
        print(f"  adapter-path: {args.adapter_path}")
    print()

    config = _load_config(args.config)

    inspect_tokenizer(config)
    inspect_model_architecture(config)

    if args.no_weights:
        _section("Model Parameters")
        _kv("skipped", "--no-weights flag set; parameter counting skipped")
    else:
        try:
            inspect_model_params(config)
        except Exception as exc:
            _section("Model Parameters")
            _kv("ERROR", str(exc))

    if args.adapter_path:
        try:
            inspect_adapter(config, args.adapter_path)
        except Exception as exc:
            _section("Adapter Inspection")
            _kv("ERROR", str(exc))

    print("\nDone.")


if __name__ == "__main__":
    main()
