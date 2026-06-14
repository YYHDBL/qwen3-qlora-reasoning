"""训练 token 分析和模型评估共享的 prompt 格式。"""

from __future__ import annotations


ANSWER_MARKER = "Answer:"


def format_evaluation_prompt(prompt: str) -> str:
    """返回模型生成时使用的精确 prompt。"""
    return f"{prompt}\n\n{ANSWER_MARKER}"


def format_training_text(
    prompt: str,
    answer: str,
    eos_token: str,
) -> tuple[str, str, str]:
    """返回用于监督微调的 prompt、answer 和完整文本。"""
    if not eos_token:
        raise ValueError("tokenizer EOS token is required")
    prompt_text = f"{format_evaluation_prompt(prompt)}\n"
    answer_text = f"{answer}{eos_token}"
    return prompt_text, answer_text, prompt_text + answer_text
