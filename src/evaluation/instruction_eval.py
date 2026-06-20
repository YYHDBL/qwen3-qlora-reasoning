#!/usr/bin/env python3
"""Stage 1 和 Stage 2 的确定性指令跟随评估。"""

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

try:
    from tqdm import tqdm as tqdm_cls
except ImportError:
    tqdm_cls = None


VALIDATOR_TYPES = {"exact", "regex", "json", "line_count", "contains"}


def validate_instruction_split_access(split: str, allow_test: bool) -> None:
    if split not in {"dev", "test"}:
        raise ValueError("instruction split must be dev or test")
    if split == "test" and not allow_test:
        raise ValueError("the frozen instruction test split requires --allow-test")


def _validate_record(
    value: Mapping[str, Any], path: Path, line_number: int
) -> dict[str, Any]:
    """校验一条评估记录的必要字段和类型合法性。"""
    # 检查必填字段是否存在（id、category、messages、validator 缺一不可）
    required = {"id", "category", "messages", "validator"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path}:{line_number} missing fields: {', '.join(missing)}")
    # id 必须为非空字符串，用于唯一标识和 error case 溯源
    if not isinstance(value["id"], str) or not value["id"]:
        raise ValueError(f"{path}:{line_number} invalid id")
    # category 用于按指令类别分组统计（如 format、constraint、multi-turn 等）
    if not isinstance(value["category"], str) or not value["category"]:
        raise ValueError(f"{path}:{line_number} invalid category")
    messages = value["messages"]
    # messages 必须为非空列表，表示多轮对话消息序列
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{line_number} messages must be nonempty")
    for index, message in enumerate(messages):
        # 每条消息必须为恰好拥有 role 和 content 两个键的字典（严格拒绝额外字段）
        if not isinstance(message, dict) or set(message) != {
            "role",
            "content",
        }:
            raise ValueError(f"{path}:{line_number} invalid message at index {index}")
        # role 限制为 system / user / assistant 三种标准角色
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError(f"{path}:{line_number} unsupported message role")
        # content 必须为纯字符串，拒绝多模态列表或其他类型
        if not isinstance(message["content"], str):
            raise ValueError(f"{path}:{line_number} message content must be a string")
    validator = value["validator"]
    # validator 为字典，type 字段决定后续使用哪种校验策略
    if not isinstance(validator, dict):
        raise ValueError(f"{path}:{line_number} validator must be an object")
    if validator.get("type") not in VALIDATOR_TYPES:
        raise ValueError(f"{path}:{line_number} unsupported validator")
    # 仅返回校验后的必要字段，丢弃 JSONL 中的冗余扩展字段
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
    # 逐行解析 JSONL 文件，跳过空白行和无效 JSON
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            # 每行必须是合法的 JSON 对象
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number} contains invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number} must contain an object")
            # 对每条记录执行字段和类型校验
            row = _validate_record(value, source, line_number)
            # 校验 id 唯一性，防止重复数据导致统计偏差
            if row["id"] in seen_ids:
                raise ValueError(f"{source}:{line_number} duplicate id: {row['id']}")
            seen_ids.add(row["id"])
            # 注入 split 元信息，便于后续按分区分析评估结果
            row["split"] = split
            rows.append(row)
    if not rows:
        raise ValueError(f"{source} contains no evaluation records")
    return rows


def _run_validator(validator: Mapping[str, Any], prediction: str) -> bool:
    """对生成的预测字符串执行单条校验规则。"""
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
        expected = validator.get("value")
        if expected is None:
            return True
        return parsed == expected
    if validator_type == "line_count":
        count = validator.get("count")
        return (
            isinstance(count, int)
            and len(prediction.splitlines()) == count
            and all(line.strip() for line in prediction.splitlines())
        )
    if validator_type == "contains":
        values = validator.get("values")
        if not (isinstance(values, list) and all(
            isinstance(value, str) and value in prediction for value in values
        )):
            return False
        max_words = validator.get("max_words")
        if isinstance(max_words, int) and len(prediction.split()) > max_words:
            return False
        max_lines = validator.get("max_lines")
        if isinstance(max_lines, int) and len(prediction.splitlines()) > max_lines:
            return False
        return True
    raise ValueError(f"unsupported validator: {validator_type}")


def evaluate_instruction_prediction(
    record: Mapping[str, Any],
    prediction_raw: str,
    stop_reason: str,
    generated_tokens: int,
) -> dict[str, Any]:
    # 根据记录的 validator 配置执行校验规则（exact / regex / json 等）
    success = _run_validator(record["validator"], prediction_raw)
    # 检查模型是否因生成 EOS 或 IM_END 而自然停止，而非因 max_length 耗尽被截断
    stop_success = stop_reason in {"im_end", "endoftext"}
    validator = record["validator"]
    # continuation_failure：模型在给出正确答案后继续生成了额外内容（"幻觉续写"）
    continuation_failure = not stop_success
    # 对 exact 校验器额外检测：如果预测以标准答案开头但更长，说明模型在
    # 正确答案之后还生成了多余 token，也标记为 continuation_failure
    if validator["type"] == "exact":
        expected = validator.get("value")
        continuation_failure = continuation_failure or (
            isinstance(expected, str)
            and prediction_raw.startswith(expected)
            and prediction_raw != expected
        )
    # 返回统一的预测评估记录，多指标便于后续分层分析（按 validator_type、category 等）
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
    """聚合一组预测的指令跟踪指标。"""
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
    # 按 category 分组，用于计算每个指令类别（format / constraint 等）的独立指标
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        groups[str(prediction["category"])].append(prediction)
    # overall：全量样本的聚合指标（微平均）
    # by_category：按类别名排序后的分类别指标，便于发现模型在哪类指令上薄弱
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
    # 先汇总指标（overall + by_category），再写入产物文件
    metrics = summarize_instruction_predictions(predictions)
    # predictions.jsonl：完整预测结果列表，每行一条 JSON 记录
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    # error_cases.jsonl：仅写入失败的预测样本，用于 badcase 分析
    # test 分区的错误样例默认不写入（保护测试数据），除非显式开启 include_test_errors
    if include_error_cases:
        write_jsonl(
            output_dir / "error_cases.jsonl",
            [row for row in predictions if not row["instruction_success"]],
        )
    # metrics.json：按 overall + by_category 的结构化指标，便于横向对比
    write_json(output_dir / "metrics.json", metrics)
    # run_config.json：运行配置快照（模型 ID、数据集哈希、生成参数等），用于实验复现
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
    _iter = zip(rows, generated, strict=True)
    if tqdm_cls is not None:
        _iter = tqdm_cls(
            _iter, total=len(rows), desc="Scoring", unit="sample",
        )
    predictions = [
        {
            **evaluate_instruction_prediction(
                row,
                result["text"],
                result["stop_reason"],
                result["generated_tokens"],
            ),
            "text_with_special": result.get("text_with_special"),
            "raw_token_ids": result.get("raw_token_ids"),
            "prompt_text": result.get("prompt_text"),
        }
        for row, result in _iter
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
