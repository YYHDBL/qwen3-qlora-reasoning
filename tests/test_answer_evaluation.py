from decimal import Decimal

import pytest

from src.evaluation.answer_evaluation import evaluate_answer


def test_bit_manipulation_requires_exact_eight_bits():
    result = evaluate_answer(
        "bit_manipulation", "10010111", "Final answer: 10010111"
    )

    assert result["parsed_answer"] == "10010111"
    assert result["parse_success"] is True
    assert result["format_valid"] is True
    assert result["strict_correct"] is True
    assert result["normalized_correct"] is True
    assert result["primary_correct"] is True


def test_gravity_uses_decimal_for_primary_metric():
    result = evaluate_answer("gravity", "57.0", "57.00")

    assert result["strict_correct"] is False
    assert result["normalized_correct"] is True
    assert result["primary_correct"] is True
    assert Decimal(result["parsed_answer"]) == Decimal("57.0")


def test_unit_conversion_requires_exactly_two_decimal_places():
    result = evaluate_answer("unit_conversion", "16.65", "16.650")

    assert result["parse_success"] is True
    assert result["format_valid"] is False
    assert result["strict_correct"] is False
    assert result["normalized_correct"] is True
    assert result["primary_correct"] is False


def test_numeral_auxiliary_metric_ignores_case_only():
    result = evaluate_answer("numeral", "XXXVIII", "xxxviii")

    assert result["strict_correct"] is False
    assert result["normalized_correct"] is True
    assert result["primary_correct"] is False


def test_cipher_auxiliary_metric_lowercases_and_collapses_spaces():
    result = evaluate_answer("cipher", "hello world", "Hello  World")

    assert result["strict_correct"] is False
    assert result["normalized_correct"] is True
    assert result["primary_correct"] is False


@pytest.mark.parametrize(
    ("gold", "prediction"),
    [
        ("\\#", "Final answer: \\#"),
        ("\"'", "Answer: \"'"),
        ("[](", "[]("),
        ("`>%/", "`>%/"),
    ],
)
def test_symbolic_transform_preserves_every_character(gold, prediction):
    result = evaluate_answer("symbolic_transform", gold, prediction)

    assert result["parsed_answer"] == gold
    assert result["strict_correct"] is True
    assert result["normalized_correct"] is True
    assert result["primary_correct"] is True


def test_extra_explanation_is_parseable_but_not_valid_output_format():
    result = evaluate_answer(
        "gravity",
        "57.0",
        "I calculated the value.\nFinal answer: 57.0",
    )

    assert result["parsed_answer"] == "57.0"
    assert result["parse_success"] is True
    assert result["format_valid"] is False
    assert result["strict_correct"] is False
    assert result["normalized_correct"] is True
    assert result["primary_correct"] is False


def test_invalid_decimal_does_not_use_float_or_raise():
    result = evaluate_answer("gravity", "57.0", "not-a-number")

    assert result["parse_success"] is False
    assert result["normalized_correct"] is False
    assert result["primary_correct"] is False
    assert result["parse_error"] == "invalid gravity answer"


def test_unknown_task_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported task type"):
        evaluate_answer("unknown", "x", "x")
