#!/usr/bin/env python3
"""校验、分类并划分推理数据集。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .classifier import ID_RE, TASK_ORDER, classify_record
from .splitters import DEFAULT_SEED, SPLIT_RATIOS, _build_splits, _length_stats


def _fullmatch(pattern: re.Pattern[str], value: str) -> bool:
    return pattern.fullmatch(value) is not None


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV，必须严格包含 ``id, prompt, answer`` 三列。"""
    # 这里要求 CSV 列顺序和字段名完全固定，避免上游导出格式变化时悄悄读错列。
    # 严格比较 fieldnames 列表而非按名称取值，是因为：
    # 1) 如果上游改变了列顺序（如 prompt, id, answer），DictReader 不会报错但数据语义全错了；
    # 2) 如果多出额外列，None in row 检测会捕获（DictReader 对多余列设为 None）。
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        # 第一道防线：列名必须精确匹配 ["id", "prompt", "answer"]
        if reader.fieldnames != ["id", "prompt", "answer"]:
            raise ValueError(
                "CSV columns must be exactly: id, prompt, answer"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            # 第二道防线：如果 CSV 含有多余列（超过表头），DictReader 会
            # 把多余值映射到 None 键，通过检查 None 键来捕获列数不一致。
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
        # 校验 1：id 必须是 8 位十六进制小写字符串，不符合则数据来源可疑
        if not _fullmatch(ID_RE, row_id):
            issue_ids["invalid_id"].append(row_id)
        # 校验 2：重复的 id 意味着数据被意外复制或合并
        if row_id in seen_ids:
            issue_ids["duplicate_id"].append(row_id)
        # 校验 3：空 prompt 无法用于训练，属于无效样本
        if not row["prompt"]:
            issue_ids["empty_prompt"].append(row_id)
        # 校验 4：空 answer 也无法用于训练
        if not row["answer"]:
            issue_ids["empty_answer"].append(row_id)
        # 校验 5：完全相同的 prompt 出现多次暗示数据去重失败
        if row["prompt"] in seen_prompts:
            issue_ids["duplicate_prompt"].append(row_id)
        # 校验 6：prompt 首尾有空白字符会干扰前缀匹配分类（classifier 做精确字符串比较）
        if row["prompt"] != row["prompt"].strip():
            issue_ids["prompt_outer_whitespace"].append(row_id)
        # 校验 7：answer 首尾空白会被误判为答案格式不匹配
        if row["answer"] != row["answer"].strip():
            issue_ids["answer_outer_whitespace"].append(row_id)
        # 校验 8：多行 answer 不符合推理任务定义的"单行简洁答案"约定
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
    """返回 *path* 的 SHA-256 十六进制摘要。"""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """写入 JSON，带末尾换行符以保持 diff 稳定。"""
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
    """读取 CSV → 基础校验 → 任务分类 → 分片 → 落盘报告。"""
    # 主流程按"读入 -> 基础校验 -> 任务分类 -> 分片 -> 落盘报告"线性执行。

    # ── 步骤 1：读取 CSV ──
    rows = _read_csv(input_path)

    # ── 步骤 2：基础字段校验 ──
    # 检查 ID 格式、空值、重复、空白字符等问题
    basic_validation = _validate_basic_fields(rows)

    # ── 步骤 3：任务分类 ──
    # 对每条记录调用 classify_record，根据 prompt 结构判定任务类型
    # 同时统计各类型数量和记录失败原因
    classified_rows: list[dict[str, str]] = []
    unknown_records: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for row in rows:
        classification = classify_record(row["prompt"], row["answer"])
        counts[classification.task_type] += 1
        # 将 task_type 附加到原始行数据中，方便后续 split 时按类型分组
        classified_rows.append({**row, "task_type": classification.task_type})
        # 分类失败的记录单独收录，包含失败原因列表方便排查
        if classification.task_type == "unknown":
            unknown_records.append(
                {"id": row["id"], "reasons": list(classification.reasons)}
            )

    # ── 步骤 4：准备输出目录 ──
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 如果输入不是标准 raw/train.csv，就把原始输入复制一份，保证输出目录里有可追溯的源数据。
    raw_path = raw_dir / "train.csv"
    if input_path.resolve() != raw_path.resolve():
        shutil.copy2(input_path, raw_path)

    # ── 步骤 5：构建分类报告 ──
    classification_report = {
        "total": len(rows),
        "unknown": counts["unknown"],
        "counts": {task: counts[task] for task in TASK_ORDER},
        "unknown_records": unknown_records,
        "basic_validation": basic_validation,
    }

    # ── 步骤 6：早期终止检查 ──
    # 发现无法归类的样本或基础字段异常时，直接中止并写失败报告，避免生成误导性的训练集。
    if unknown_records or not basic_validation["valid"]:
        # 提前退出：不进行分片，只写入包含错误详情的报告文件
        failure_report = {
            "classification": classification_report,
            "status": "failed_validation",
        }
        _write_json(processed_dir / "dataset_report.json", failure_report)
        raise ValueError(
            "dataset validation failed; inspect dataset_report.json"
        )

    # ── 步骤 7：分片 ──
    # 调用 _build_splits 进行确定性的分层 train/val/test 划分
    splits, quotas = _build_splits(classified_rows, seed)
    output_paths = {
        split: processed_dir / f"{split}.jsonl" for split in SPLIT_RATIOS
    }

    # 先写 split 文件，再基于实际产物计算 manifest 哈希，确保索引文件与内容完全一致。
    for split, output_path in output_paths.items():
        _write_jsonl(output_path, splits[split])

    # ── 步骤 8：构建分片统计 ──
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

    # ── 步骤 9：输出 split manifest ──
    # manifest 记录了每个 split 的 meta 信息：样本数、任务分布、ID 列表、文件哈希
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

    # ── 步骤 10：按任务类型分组，用于报告中的字符长度统计 ──
    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in classified_rows:
        rows_by_task[row["task_type"]].append(row)

    # ── 步骤 11：输出完整的 dataset_report.json ──
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
