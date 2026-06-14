#!/usr/bin/env python3
"""Deterministic instruction-following evaluation for Stage 1 and Stage 2."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ..common.config import load_yaml_config, validate_stage1_config
from ..common.experiment import (
    package_versions,
    sha256_file,
    write_json,
    write_jsonl,
)


VALIDATOR_TYPES = {"exact", "regex", "json", "line_count", "contains"}


def validate_instruction_split_access(split: str, allow_test: bool) -> None:
    if split not in {"dev", "test"}:
        raise ValueError("instruction split must be dev or test")
    if split == "test" and not allow_test:
        raise ValueError("the frozen instruction test split requires --allow-test")


def _validate_record(
    value: Mapping[str, Any], path: Path, line_number: int
) -> dict[str, Any]:
    required = {"id", "category", "messages", "validator"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path}:{line_number} missing fields: {', '.join(missing)}")
    if not isinstance(value["id"], str) or not value["id"]:
        raise ValueError(f"{path}:{line_number} invalid id")
    if not isinstance(value["category"], str) or not value["category"]:
        raise ValueError(f"{path}:{line_number} invalid category")
    messages = value["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{line_number} messages must be nonempty")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {
            "role",
            "content",
        }:
            raise ValueError(f"{path}:{line_number} invalid message at index {index}")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError(f"{path}:{line_number} unsupported message role")
        if not isinstance(message["content"], str):
            raise ValueError(f"{path}:{line_number} message content must be a string")
    validator = value["validator"]
    if not isinstance(validator, dict):
        raise ValueError(f"{path}:{line_number} validator must be an object")
    if validator.get("type") not in VALIDATOR_TYPES:
        raise ValueError(f"{path}:{line_number} unsupported validator")
    return {
        "id": value["id"],
        "category": value["category"],
        "messages": messages,
        "validator": validator,
    }


def load_instruction_eval(path: str | Path, split: str) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number} contains invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number} must contain an object")
            row = _validate_record(value, source, line_number)
            if row["id"] in seen_ids:
                raise ValueError(f"{source}:{line_number} duplicate id: {row['id']}")
            seen_ids.add(row["id"])
            row["split"] = split
            rows.append(row)
    if not rows:
        raise ValueError(f"{source} contains no evaluation records")
    return rows


def _run_validator(validator: Mapping[str, Any], prediction: str) -> bool:
    validator_type = validator["type"]
    if validator_type == "exact":
        return prediction == validator.get("value")
    if validator_type == "regex":
        pattern = validator.get("pattern")
        return (
            isinstance(pattern, str) and re.fullmatch(pattern, prediction) is not None
        )
    if validator_type == "json":
        try:
            parsed = json.loads(prediction)
        except json.JSONDecodeError:
            return False
        return parsed == validator.get("value")
    if validator_type == "line_count":
        count = validator.get("count")
        return (
            isinstance(count, int)
            and len(prediction.splitlines()) == count
            and all(line.strip() for line in prediction.splitlines())
        )
    if validator_type == "contains":
        values = validator.get("values")
        return isinstance(values, list) and all(
            isinstance(value, str) and value in prediction for value in values
        )
    raise ValueError(f"unsupported validator: {validator_type}")


def evaluate_instruction_prediction(
    record: Mapping[str, Any],
    prediction_raw: str,
    stop_reason: str,
    generated_tokens: int,
) -> dict[str, Any]:
    success = _run_validator(record["validator"], prediction_raw)
    stop_success = stop_reason in {"im_end", "endoftext"}
    validator = record["validator"]
    continuation_failure = not stop_success
    if validator["type"] == "exact":
        expected = validator.get("value")
        continuation_failure = continuation_failure or (
            isinstance(expected, str)
            and prediction_raw.startswith(expected)
            and prediction_raw != expected
        )
    return {
        "id": record["id"],
        "split": record.get("split"),
        "category": record["category"],
        "prediction_raw": prediction_raw,
        "validator_type": validator["type"],
        "instruction_success": success,
        "format_success": success,
        "stop_success": stop_success,
        "stop_reason": stop_reason,
        "continuation_failure": continuation_failure,
        "generated_tokens": generated_tokens,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty instruction predictions")
    count = len(rows)
    return {
        "count": count,
        "instruction_success_count": sum(
            bool(row["instruction_success"]) for row in rows
        ),
        "instruction_accuracy": sum(bool(row["instruction_success"]) for row in rows)
        / count,
        "format_success_count": sum(bool(row["format_success"]) for row in rows),
        "format_accuracy": sum(bool(row["format_success"]) for row in rows) / count,
        "stop_success_count": sum(bool(row["stop_success"]) for row in rows),
        "stop_accuracy": sum(bool(row["stop_success"]) for row in rows) / count,
        "continuation_failure_count": sum(
            bool(row["continuation_failure"]) for row in rows
        ),
        "continuation_failure_rate": sum(
            bool(row["continuation_failure"]) for row in rows
        )
        / count,
        "mean_generated_tokens": mean(int(row["generated_tokens"]) for row in rows),
        "max_generated_tokens": max(int(row["generated_tokens"]) for row in rows),
    }


def summarize_instruction_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        groups[str(prediction["category"])].append(prediction)
    return {
        "overall": _summarize(predictions),
        "by_category": {
            category: _summarize(rows) for category, rows in sorted(groups.items())
        },
    }


def write_instruction_artifacts(
    output_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    include_error_cases: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize_instruction_predictions(predictions)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    if include_error_cases:
        write_jsonl(
            output_dir / "error_cases.jsonl",
            [row for row in predictions if not row["instruction_success"]],
        )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_config.json", run_config)
    return metrics


def run_instruction_evaluation(
    config: Mapping[str, Any],
    split: str,
    output_dir: Path,
    adapter_path: str | None,
    allow_test: bool,
    limit: int | None,
    include_test_errors: bool,
) -> dict[str, Any]:
    validate_instruction_split_access(split, allow_test)
    path = Path(config["evaluation"][f"{split}_path"])
    rows = load_instruction_eval(path, split)
    if limit is not None:
        rows = rows[:limit]

    from ..training.model_loader import ChatGenerator

    generator = ChatGenerator(config, adapter_path=adapter_path)
    generated = generator.generate(
        [row["messages"] for row in rows],
        batch_size=int(config["evaluation"]["batch_size"]),
    )
    predictions = [
        evaluate_instruction_prediction(
            row,
            result["text"],
            result["stop_reason"],
            result["generated_tokens"],
        )
        for row, result in zip(rows, generated, strict=True)
    ]
    run_config = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model"]["id"],
        "model_revision": (generator.model_revision or config["model"].get("revision")),
        "tokenizer_revision": generator.tokenizer_revision,
        "adapter_path": adapter_path,
        "split": split,
        "dataset_path": str(path),
        "dataset_sha256": sha256_file(path),
        "count": len(rows),
        "generation": config["generation"],
        "packages": package_versions(["torch", "transformers", "peft", "accelerate"]),
    }
    return write_instruction_artifacts(
        output_dir,
        predictions,
        run_config,
        include_error_cases=(split == "dev" or include_test_errors),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Base or LoRA instruction following."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_no_robots.yaml"),
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--include-test-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_stage1_config(config)
    metrics = run_instruction_evaluation(
        config,
        split=args.split,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        allow_test=args.allow_test,
        limit=args.limit,
        include_test_errors=args.include_test_errors,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
