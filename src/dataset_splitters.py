from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

try:
    from src.dataset_classifier import TASK_ORDER
except ModuleNotFoundError:
    from dataset_classifier import TASK_ORDER


DEFAULT_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _tie_break(seed: int, name: str) -> str:
    value = f"{seed}:{name}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _apportion(
    counts: Mapping[str, int],
    ratio: float,
    target_total: int,
    seed: int,
) -> dict[str, int]:
    raw = {task: count * ratio for task, count in counts.items()}
    allocation = {task: math.floor(value) for task, value in raw.items()}
    remaining = target_total - sum(allocation.values())
    ranked_tasks = sorted(
        counts,
        key=lambda task: (
            -(raw[task] - math.floor(raw[task])),
            _tie_break(seed, task),
        ),
    )
    for task in ranked_tasks[:remaining]:
        allocation[task] += 1
    return allocation


def allocate_split_quotas(
    task_counts: Mapping[str, int],
    ratios: Mapping[str, float] = SPLIT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, int]]:
    """Allocate exact global split totals while preserving task proportions."""
    if set(ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must contain train, validation, and test")
    if not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split ratios must sum to 1")

    total = sum(task_counts.values())
    train_total = round(total * ratios["train"])
    validation_total = round(total * ratios["validation"])

    train = _apportion(
        task_counts, ratios["train"], train_total, seed=seed
    )
    validation = _apportion(
        task_counts,
        ratios["validation"],
        validation_total,
        seed=seed + 1,
    )
    test = {
        task: task_counts[task] - train[task] - validation[task]
        for task in task_counts
    }
    if any(count < 0 for count in test.values()):
        raise ValueError("split allocation produced a negative test quota")

    return {"train": train, "validation": validation, "test": test}


def _build_splits(
    rows: Sequence[dict[str, str]],
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_task[row["task_type"]].append(row)

    task_counts = {
        task: len(rows_by_task[task])
        for task in TASK_ORDER
        if rows_by_task[task]
    }
    quotas = allocate_split_quotas(task_counts, seed=seed)
    splits: dict[str, list[dict[str, str]]] = {
        split: [] for split in SPLIT_RATIOS
    }

    for task_index, task in enumerate(TASK_ORDER):
        task_rows = list(rows_by_task[task])
        random.Random(seed + task_index).shuffle(task_rows)
        train_end = quotas["train"].get(task, 0)
        validation_end = train_end + quotas["validation"].get(task, 0)
        splits["train"].extend(task_rows[:train_end])
        splits["validation"].extend(task_rows[train_end:validation_end])
        splits["test"].extend(task_rows[validation_end:])

    for split_index, split in enumerate(SPLIT_RATIOS):
        random.Random(seed + 100 + split_index).shuffle(splits[split])

    return splits, quotas


def _percentile(sorted_values: Sequence[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    index = min(
        len(sorted_values) - 1,
        math.ceil(percentile * len(sorted_values)) - 1,
    )
    return sorted_values[index]


def _length_stats(values: Iterable[str]) -> dict[str, float | int]:
    lengths = sorted(len(value) for value in values)
    if not lengths:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p90": 0,
            "p99": 0,
        }
    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(mean(lengths), 2),
        "median": median(lengths),
        "p90": _percentile(lengths, 0.90),
        "p99": _percentile(lengths, 0.99),
    }
