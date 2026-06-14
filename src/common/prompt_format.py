"""Shared prompt formats for training analysis and model evaluation."""

from __future__ import annotations


ANSWER_MARKER = "Answer:"


def format_evaluation_prompt(prompt: str) -> str:
    """Return the exact prompt used for model generation."""
    return f"{prompt}\n\n{ANSWER_MARKER}"


def format_training_text(
    prompt: str,
    answer: str,
    eos_token: str,
) -> tuple[str, str, str]:
    """Return prompt, answer, and full text for supervised fine-tuning."""
    if not eos_token:
        raise ValueError("tokenizer EOS token is required")
    prompt_text = f"{format_evaluation_prompt(prompt)}\n"
    answer_text = f"{answer}{eos_token}"
    return prompt_text, answer_text, prompt_text + answer_text
