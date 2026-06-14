#!/usr/bin/env python3
"""Validate, classify, and split the reasoning dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

try:
    from src.dataset_classifier import (
        ID_RE,
        TASK_ORDER,
        classify_record,
    )
    from src.dataset_splitters import (
        DEFAULT_SEED,
        SPLIT_RATIOS,
        _build_splits,
        _length_stats,
    )
except ModuleNotFoundError:
    from dataset_classifier import (
        ID_RE,
        TASK_ORDER,
        classify_record,
    )
    from dataset_splitters import (
        DEFAULT_SEED,
        SPLIT_RATIOS,
        _build_splits,
        _length_stats,
    )


def _fullmatch(pattern: re.Pattern[str], value: str) -> bool:
    return pattern.fullmatch(value) is not None


def _read_csv(path: Path) -> list[dict[str, str]]:
    # 这里要求 CSV 列顺序和字段名完全固定，避免上游导出格式变化时悄悄读错列。
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "prompt", "answer"]:
            raise ValueError(
                "CSV columns must be exactly: id, prompt, answer"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"row {line_number} has extra CSV fields")
            rows.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                }
            )
    return rows


def _validate_basic_fields(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    # 基础校验只关注与业务无关的通用数据质量问题：ID、空值、重复、空白和多行答案。
    # 这些问题一旦存在，后续分类和切分结果就不可信。
    issue_ids: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()

    for row in rows:
        row_id = row["id"]
        if not _fullmatch(ID_RE, row_id):
            issue_ids["invalid_id"].append(row_id)
        if row_id in seen_ids:
            issue_ids["duplicate_id"].append(row_id)
        if not row["prompt"]:
            issue_ids["empty_prompt"].append(row_id)
        if not row["answer"]:
            issue_ids["empty_answer"].append(row_id)
        if row["prompt"] in seen_prompts:
            issue_ids["duplicate_prompt"].append(row_id)
        if row["prompt"] != row["prompt"].strip():
            issue_ids["prompt_outer_whitespace"].append(row_id)
        if row["answer"] != row["answer"].strip():
            issue_ids["answer_outer_whitespace"].append(row_id)
        if "\n" in row["answer"] or "\r" in row["answer"]:
            issue_ids["multiline_answer"].append(row_id)
        seen_ids.add(row_id)
        seen_prompts.add(row["prompt"])

    return {
        "valid": not issue_ids,
        "counts": {
            issue: len(ids) for issue, ids in sorted(issue_ids.items())
        },
        "ids": {issue: ids for issue, ids in sorted(issue_ids.items())},
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    # 统一使用 UTF-8 + 末尾换行，保证产物在不同平台上都稳定可读、可 diff。
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    # JSONL 里显式写出 task_type，方便训练阶段直接消费，无需再从 prompt 反推类别。
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            record = {
                "id": row["id"],
                "task_type": row["task_type"],
                "prompt": row["prompt"],
                "answer": row["answer"],
            }
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")


def prepare_dataset(input_path: Path, data_dir: Path, seed: int) -> None:
    # 主流程按“读入 -> 基础校验 -> 任务分类 -> 分片 -> 落盘报告”线性执行。
    rows = _read_csv(input_path)
    basic_validation = _validate_basic_fields(rows)

    classified_rows: list[dict[str, str]] = []
    unknown_records: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for row in rows:
        classification = classify_record(row["prompt"], row["answer"])
        counts[classification.task_type] += 1
        classified_rows.append({**row, "task_type": classification.task_type})
        if classification.task_type == "unknown":
            unknown_records.append(
                {"id": row["id"], "reasons": list(classification.reasons)}
            )

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # 如果输入不是标准 raw/train.csv，就把原始输入复制一份，保证输出目录里有可追溯的源数据。
    raw_path = raw_dir / "train.csv"
    if input_path.resolve() != raw_path.resolve():
        shutil.copy2(input_path, raw_path)

    classification_report = {
        "total": len(rows),
        "unknown": counts["unknown"],
        "counts": {task: counts[task] for task in TASK_ORDER},
        "unknown_records": unknown_records,
        "basic_validation": basic_validation,
    }

    if unknown_records or not basic_validation["valid"]:
        # 发现无法归类的样本或基础字段异常时，直接中止并写失败报告，避免生成误导性的训练集。
        failure_report = {
            "classification": classification_report,
            "status": "failed_validation",
        }
        _write_json(processed_dir / "dataset_report.json", failure_report)
        raise ValueError(
            "dataset validation failed; inspect dataset_report.json"
        )

    splits, quotas = _build_splits(classified_rows, seed)
    output_paths = {
        split: processed_dir / f"{split}.jsonl" for split in SPLIT_RATIOS
    }
    # 先写 split 文件，再基于实际产物计算 manifest 哈希，确保索引文件与内容完全一致。
    for split, output_path in output_paths.items():
        _write_jsonl(output_path, splits[split])

    split_counts = {
        split: {
            "total": len(split_rows),
            "by_task_type": {
                task: sum(
                    row["task_type"] == task for row in split_rows
                )
                for task in TASK_ORDER
            },
        }
        for split, split_rows in splits.items()
    }
    manifest = {
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "source": {
            "path": "data/raw/train.csv",
            "sha256": _sha256(raw_path),
            "total": len(rows),
        },
        "splits": {
            split: {
                **split_counts[split],
                "ids": [row["id"] for row in splits[split]],
                "sha256": _sha256(output_paths[split]),
            }
            for split in SPLIT_RATIOS
        },
        "task_quotas": quotas,
    }
    _write_json(processed_dir / "split_manifest.json", manifest)

    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in classified_rows:
        rows_by_task[row["task_type"]].append(row)
    # 这是最终的汇总报告：既包括分类结果，也包括每个 split 的统计分布和字符长度分布。
    report = {
        "status": "ok",
        "classification": classification_report,
        "splits": split_counts,
        "character_lengths": {
            "overall": {
                "prompt": _length_stats(
                    row["prompt"] for row in classified_rows
                ),
                "answer": _length_stats(
                    row["answer"] for row in classified_rows
                ),
            },
            "by_task_type": {
                task: {
                    "prompt": _length_stats(
                        row["prompt"] for row in rows_by_task[task]
                    ),
                    "answer": _length_stats(
                        row["answer"] for row in rows_by_task[task]
                    ),
                }
                for task in TASK_ORDER
            },
        },
        "files": {
            "raw": {
                "path": "data/raw/train.csv",
                "sha256": _sha256(raw_path),
            },
            **{
                split: {
                    "path": f"data/processed/{split}.jsonl",
                    "sha256": _sha256(output_paths[split]),
                }
                for split in SPLIT_RATIOS
            },
            "split_manifest": {
                "path": "data/processed/split_manifest.json"
            },
        },
        "token_statistics": {
            "status": "deferred",
            "reason": (
                "Token counts must be computed with the "
                "Qwen/Qwen3-4B-Base tokenizer."
            ),
        },
    }
    _write_json(processed_dir / "dataset_report.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, classify, and split train.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train.csv"),
        help="Source CSV path (default: train.csv).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Output data directory (default: data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic split seed (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(args.input, args.data_dir, args.seed)


if __name__ == "__main__":
    main()
