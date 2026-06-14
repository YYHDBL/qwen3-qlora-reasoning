"""Qwen3 Chat Template helpers with delayed TRL imports."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


IM_END_TOKEN = "<|im_end|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"


def configure_training_chat_template(tokenizer: Any) -> str:
    """Attach TRL's output-compatible Qwen3 training template."""
    from trl.chat_template_utils import get_training_chat_template

    template = get_training_chat_template(processing_class=tokenizer)
    if template is not None:
        tokenizer.chat_template = template
    active_template = tokenizer.get_chat_template()
    if "{% generation" not in active_template and "{%- generation" not in (
        active_template
    ):
        raise ValueError("training chat template has no generation markers")
    tokenizer._stage1_training_chat_template = active_template
    return active_template


def render_generation_prompt(
    tokenizer: Any, messages: Sequence[Mapping[str, str]]
) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def resolve_stop_token_ids(tokenizer: Any) -> dict[str, int]:
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    eos_id = tokenizer.eos_token_id
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if im_end_id is None or im_end_id == unk_id:
        raise ValueError(f"tokenizer does not define {IM_END_TOKEN}")
    if eos_id is None:
        raise ValueError("tokenizer does not define an EOS token")
    return {"im_end": int(im_end_id), "endoftext": int(eos_id)}
