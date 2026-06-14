import json

from src.evaluation_metrics import compute_metrics


def test_metrics_include_overall_and_per_task_type():
    predictions = [
        {
            "task_type": "gravity",
            "parse_success": True,
            "format_valid": True,
            "primary_correct": True,
            "strict_correct": False,
            "normalized_correct": True,
        },
        {
            "task_type": "gravity",
            "parse_success": False,
            "format_valid": False,
            "primary_correct": False,
            "strict_correct": False,
            "normalized_correct": False,
        },
        {
            "task_type": "cipher",
            "parse_success": True,
            "format_valid": True,
            "primary_correct": True,
            "strict_correct": True,
            "normalized_correct": True,
        },
    ]

    metrics = compute_metrics(predictions)

    assert metrics["overall"] == {
        "count": 3,
        "parse_success_count": 2,
        "parse_success_rate": 2 / 3,
        "format_valid_count": 2,
        "format_valid_rate": 2 / 3,
        "primary_correct_count": 2,
        "primary_accuracy": 2 / 3,
        "strict_correct_count": 1,
        "strict_accuracy": 1 / 3,
        "normalized_correct_count": 2,
        "normalized_accuracy": 2 / 3,
        "error_count": 1,
    }
    assert metrics["by_task_type"]["gravity"]["count"] == 2
    assert metrics["by_task_type"]["gravity"]["primary_accuracy"] == 0.5
    assert metrics["by_task_type"]["cipher"]["primary_accuracy"] == 1.0
    assert metrics["macro_primary_accuracy"] == 0.75
    assert json.dumps(metrics)


def test_metrics_reject_empty_predictions():
    try:
        compute_metrics([])
    except ValueError as exc:
        assert str(exc) == "cannot compute metrics for empty predictions"
    else:
        raise AssertionError("empty predictions must fail")
