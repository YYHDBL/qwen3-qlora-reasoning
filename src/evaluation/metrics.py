"""Aggregate answer evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    parse_success_count = sum(bool(r["parse_success"]) for r in records)
    format_valid_count = sum(bool(r["format_valid"]) for r in records)
    primary_correct_count = sum(bool(r["primary_correct"]) for r in records)
    strict_correct_count = sum(bool(r["strict_correct"]) for r in records)
    normalized_correct_count = sum(
        bool(r["normalized_correct"]) for r in records
    )
    return {
        "count": count,
        "parse_success_count": parse_success_count,
        "parse_success_rate": parse_success_count / count,
        "format_valid_count": format_valid_count,
        "format_valid_rate": format_valid_count / count,
        "primary_correct_count": primary_correct_count,
        "primary_accuracy": primary_correct_count / count,
        "strict_correct_count": strict_correct_count,
        "strict_accuracy": strict_correct_count / count,
        "normalized_correct_count": normalized_correct_count,
        "normalized_accuracy": normalized_correct_count / count,
        "error_count": count - primary_correct_count,
    }


def compute_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot compute metrics for empty predictions")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        groups[str(prediction["task_type"])].append(prediction)
    by_task_type = {
        task_type: _summarize(records)
        for task_type, records in sorted(groups.items())
    }
    return {
        "overall": _summarize(predictions),
        "by_task_type": by_task_type,
        "macro_primary_accuracy": mean(
            metrics["primary_accuracy"]
            for metrics in by_task_type.values()
        ),
    }
