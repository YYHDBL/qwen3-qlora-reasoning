from src.prompt_format import (
    format_evaluation_prompt,
    format_training_text,
)


def test_evaluation_prompt_uses_exact_shared_template():
    prompt = "  Preserve this prompt exactly.  "

    result = format_evaluation_prompt(prompt)

    assert result == "  Preserve this prompt exactly.  \n\nAnswer:"
    assert prompt == "  Preserve this prompt exactly.  "


def test_training_text_builds_on_evaluation_prompt():
    prompt_text, answer_text, full_text = format_training_text(
        "Question?", "42", "<eos>"
    )

    assert prompt_text == "Question?\n\nAnswer:\n"
    assert answer_text == "42<eos>"
    assert full_text == "Question?\n\nAnswer:\n42<eos>"


def test_training_text_requires_eos():
    try:
        format_training_text("Question?", "42", "")
    except ValueError as exc:
        assert str(exc) == "tokenizer EOS token is required"
    else:
        raise AssertionError("missing EOS token must fail")
