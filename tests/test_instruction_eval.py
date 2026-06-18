import json

import pytest

from src.evaluation.instruction_eval import (
    evaluate_instruction_prediction,
    load_instruction_eval,
    summarize_instruction_predictions,
    validate_instruction_split_access,
)


@pytest.mark.parametrize(
    ("validator", "prediction", "expected"),
    [
        ({"type": "exact", "value": "BLUE"}, "BLUE", True),
        ({"type": "regex", "pattern": r"^[A-Z]{3}$"}, "ABC", True),
        (
            {"type": "json", "value": {"answer": 4}},
            '{"answer": 4}',
            True,
        ),
        ({"type": "line_count", "count": 2}, "one\ntwo", True),
        (
            {"type": "contains", "values": ["alpha", "beta"]},
            "beta then alpha",
            True,
        ),
    ],
)
def test_instruction_validators(validator, prediction, expected):
    result = evaluate_instruction_prediction(
        {"id": "x", "category": "format", "validator": validator},
        prediction,
        stop_reason="im_end",
        generated_tokens=3,
    )

    assert result["instruction_success"] is expected
    assert result["stop_success"] is True


def test_exact_validator_rejects_extra_continuation():
    result = evaluate_instruction_prediction(
        {
            "id": "x",
            "category": "stop",
            "validator": {"type": "exact", "value": "DONE"},
        },
        "DONE\nHere is an explanation.",
        stop_reason="length",
        generated_tokens=20,
    )

    assert result["instruction_success"] is False
    assert result["format_success"] is False
    assert result["stop_success"] is False
    assert result["continuation_failure"] is True


def test_instruction_metrics_include_overall_and_categories():
    predictions = [
        {
            "category": "format",
            "instruction_success": True,
            "format_success": True,
            "stop_success": True,
            "continuation_failure": False,
            "generated_tokens": 2,
        },
        {
            "category": "format",
            "instruction_success": False,
            "format_success": False,
            "stop_success": False,
            "continuation_failure": True,
            "generated_tokens": 8,
        },
    ]

    metrics = summarize_instruction_predictions(predictions)

    assert metrics["overall"]["instruction_accuracy"] == 0.5
    assert metrics["overall"]["mean_generated_tokens"] == 5
    assert metrics["by_category"]["format"]["count"] == 2
    assert json.dumps(metrics)


def test_instruction_test_requires_explicit_access():
    with pytest.raises(ValueError, match="frozen instruction test"):
        validate_instruction_split_access("test", allow_test=False)

    validate_instruction_split_access("dev", allow_test=False)
    validate_instruction_split_access("test", allow_test=True)


def test_repository_eval_files_have_disjoint_ids():
    dev = load_instruction_eval("data/eval/instruction_dev.jsonl", "dev")
    test = load_instruction_eval("data/eval/instruction_test.jsonl", "test")

    assert dev
    assert test
    assert {row["id"] for row in dev}.isdisjoint({row["id"] for row in test})


def test_contains_validator_rejects_if_exceeding_max_words():
    result = evaluate_instruction_prediction(
        {
            "id": "x",
            "category": "content_constraint",
            "validator": {"type": "contains", "values": ["rain", "glass"], "max_words": 5},
        },
        "rain falls on the glass window pane",
        stop_reason="im_end",
        generated_tokens=7,
    )

    assert result["instruction_success"] is False


def test_contains_validator_passes_within_word_limit():
    result = evaluate_instruction_prediction(
        {
            "id": "x",
            "category": "content_constraint",
            "validator": {"type": "contains", "values": ["rain", "glass"], "max_words": 8},
        },
        "rain on the glass",
        stop_reason="im_end",
        generated_tokens=4,
    )

    assert result["instruction_success"] is True


def test_contains_validator_rejects_if_exceeding_max_lines():
    result = evaluate_instruction_prediction(
        {
            "id": "x",
            "category": "content_constraint",
            "validator": {"type": "contains", "values": ["alpha"], "max_lines": 1},
        },
        "alpha\nbeta",
        stop_reason="im_end",
        generated_tokens=2,
    )

    assert result["instruction_success"] is False


def test_contains_validator_without_limits_still_works():
    result = evaluate_instruction_prediction(
        {
            "id": "x",
            "category": "content_constraint",
            "validator": {"type": "contains", "values": ["hello", "world"]},
        },
        "hello beautiful world and many more words here",
        stop_reason="im_end",
        generated_tokens=8,
    )

    assert result["instruction_success"] is True
