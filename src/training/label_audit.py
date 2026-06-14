#!/usr/bin/env python3
"""在模型训练前审计 Qwen3 assistant-only 标签。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..common.config import load_yaml_config, validate_stage1_config
from ..common.experiment import sha256_file, write_json
from ..evaluation.analyze_tokens import summarize_values
from .chat_template import (
    IM_END_TOKEN,
    configure_training_chat_template,
)


def _assistant_mask(rendered: Mapping[str, Any]) -> list[int]:
    """从 tokenizer 输出中提取二进制 assistant mask。

    TRL 训练模板返回 ``assistant_tokens_mask``（1 = 监督信号, 0 = 忽略）。
    部分模板版本使用旧键名 ``assistant_masks``，此函数兼容两者。
    """
    value = rendered.get("assistant_masks")
    if value is None:
        value = rendered.get("assistant_tokens_mask")
    if value is None:
        raise ValueError("chat template did not return an assistant token mask")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one conversation during label audit")
        value = value[0]
    return [int(item) for item in value]


def _supervised_spans(mask: Sequence[int]) -> list[tuple[int, int]]:
    """将二进制 mask 转换为 ``(start, end)`` 区间列表。"""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _span_boundary_checks(
    spans: Sequence[tuple[int, int]],
    input_ids: Sequence[int],
    im_end_id: int,
) -> bool:
    """检查每个完整 assistant span 后是否有一个 <|im_end|> 边界标记。"""
    for start, end in spans:
        if im_end_id in input_ids[start : end + 1]:
            continue
        boundary_index = end + 1
        if boundary_index >= len(input_ids) or input_ids[boundary_index] != im_end_id:
            return False
    return True


def audit_conversation(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    # ── 步骤 1: tokenize 聊天记录 ──
    # apply_chat_template: 用训练 chat template 渲染对话为 token 序列
    #   - return_assistant_tokens_mask=True: 同时返回 assistant 位置掩码
    #   - add_generation_prompt=False: 不追加 "<|im_start|>assistant\n"（审计时不需要生成 prompt）
    rendered = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    input_ids = rendered["input_ids"]
    attention_mask = rendered.get("attention_mask", [1] * len(input_ids))
    # batch=1 时 apply_chat_template 可能返回嵌套列表，需要展平
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        attention_mask = attention_mask[0]

    # ── 步骤 2: 提取 assistant token mask ──
    # mask: 1=assistant 回复的 token（需要计算 loss），0=user/system/tool 的 token（忽略）
    mask = _assistant_mask(rendered)
    if not (len(input_ids) == len(attention_mask) == len(mask)):
        raise ValueError("tokenizer returned inconsistent mask lengths")

    # ── 步骤 3: 保存原始（未截断）数据 ──
    original_input_ids = list(input_ids)
    original_attention_mask = list(attention_mask)
    original_mask = list(mask)
    original_length = len(original_input_ids)
    # 统计原始 assistant token 数量
    original_spans = _supervised_spans(original_mask)
    original_supervised_tokens = sum(bool(value) for value in original_mask)
    if not original_supervised_tokens:
        raise ValueError(
            f"{record.get('id', '<unknown>')} has no supervised assistant tokens"
        )

    # ── 步骤 4: 验证每个 assistant span 后是否有 <|im_end|> 边界标记 ──
    # 这是训练 chat template 正确性的关键检查：
    #   - 每个 assistant 回复必须以 <|im_end|> 结尾（聊天模板语法要求）
    #   - 缺少边界标记会导致 assistant 回复与后续 user 消息粘连
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    im_end_follows_supervised_span = _span_boundary_checks(
        original_spans,
        original_input_ids,
        im_end_id,
    )
    if not im_end_follows_supervised_span:
        raise ValueError(
            f"{record.get('id', '<unknown>')} is missing the terminating "
            f"{IM_END_TOKEN} boundary after a supervised assistant span; "
            "check the training chat template"
        )

    # ── 步骤 5: 检查是否需要截断 ──
    # 序列长度超出 max_length 的样本会被标记为 ineligible_for_training
    truncated = original_length > max_length
    eligible_for_training = not truncated
    exclusion_reason = "sequence_exceeds_max_length" if truncated else None
    # 额外检查：截断位置是否恰好切在 assistant span 中间
    # 如果是，截断后 assistant 回复不完整，会导致模型学到错误的结尾模式
    truncates_supervised_span = any(
        start < max_length <= end for start, end in original_spans
    )

    # ── 步骤 6: 截断到 max_length ──
    input_ids = original_input_ids[:max_length]
    attention_mask = original_attention_mask[:max_length]
    mask = original_mask[:max_length]
    # 重新计算截断后的 assistant span
    retained_spans = _supervised_spans(mask)

    # ── 步骤 7: 构建训练 labels ──
    # labels: assistant token 保留原始 token_id，非 assistant token 设为 -100
    #   -100 是 PyTorch CrossEntropyLoss 的 ignore_index，不参与 loss 计算
    #   - 这确保模型只在 assistant 回复上学习，不在 user prompt 上产生 loss
    labels = [
        token_id if assistant else -100
        for token_id, assistant in zip(input_ids, mask, strict=True)
    ]
    # 提取所有被监督（参与 loss）的 token，用于后续解码查看
    supervised_ids = [
        token_id
        for token_id, label in zip(input_ids, labels, strict=True)
        if label != -100
    ]

    # ── 步骤 8: 额外诊断信息 ──
    # 截断后序列是否以 assistant token 结尾（可能在 loss 计算时产生边界问题）
    ends_with_supervised_token = bool(mask and mask[-1])
    # <|im_end|> token 本身是否被标记为 assistant 的一部分（应该是）
    im_end_supervised = any(
        token_id == im_end_id and assistant
        for token_id, assistant in zip(input_ids, mask, strict=True)
    )
    # 构建完整的审计记录字典
    return {
        "id": record.get("id"),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "original_tokens": original_length,
        "sequence_tokens": len(input_ids),
        "supervised_tokens": len(supervised_ids),
        "original_supervised_tokens": original_supervised_tokens,
        "masked_tokens": len(labels) - len(supervised_ids),
        "truncated": truncated,
        "eligible_for_training": eligible_for_training,
        "exclusion_reason": exclusion_reason,
        "truncates_supervised_span": truncates_supervised_span,
        "supervised_span_count": len(retained_spans),
        "supervised_spans": retained_spans,
        "original_supervised_span_count": len(original_spans),
        "ends_with_supervised_token": ends_with_supervised_token,
        "im_end_supervised": im_end_supervised,
        "im_end_follows_supervised_span": im_end_follows_supervised_span,
        # 解码后的文本用于人工检查 tokenizer 行为
        "decoded_sequence": tokenizer.decode(
            input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "decoded_supervised": tokenizer.decode(
            supervised_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def read_conversations(path: Path) -> list[dict[str, Any]]:
    """加载 JSONL 对话，校验每行包含 ``messages`` 列表。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(
                value.get("messages"), list
            ):
                raise ValueError(f"{path}:{line_number} missing conversation messages")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def audit_records(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """对 split 中的每条记录执行 ``audit_conversation``。"""
    return [audit_conversation(record, tokenizer, max_length) for record in records]


def build_audit_reports(
    audited_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    input_paths: Mapping[str, Path],
    model_id: str,
    tokenizer: Any,
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # ── token_report: 按 split 汇总的全局统计 ──
    #   - 包含 tokenizer 信息、输入文件哈希、每个 split 的样本数/合格数/排除数/排除ID
    #   - 以及 token 数量分布（original_tokens, sequence_tokens, supervised_tokens）
    token_report = {
        "model_id": model_id,
        "max_length": max_length,
        "tokenizer_class": tokenizer.__class__.__name__,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "im_end_token": IM_END_TOKEN,
        "im_end_token_id": tokenizer.convert_tokens_to_ids(IM_END_TOKEN),
        # 记录输入文件的 sha256，确保训练和审计的数据一致
        "input_files": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for split, path in input_paths.items()
        },
        "splits": {
            split: {
                "count": len(rows),
                "eligible_count": sum(
                    bool(row["eligible_for_training"]) for row in rows
                ),
                "excluded_count": sum(
                    not bool(row["eligible_for_training"]) for row in rows
                ),
                # excluded_ids 供后续训练阶段在数据加载时过滤
                "excluded_ids": [
                    row["id"]
                    for row in rows
                    if not bool(row["eligible_for_training"])
                ],
                # 统计排除原因分布（例如 "sequence_exceeds_max_length" 出现了多少次）
                "exclusion_reasons": dict(
                    sorted(
                        Counter(
                            str(row["exclusion_reason"])
                            for row in rows
                            if row["exclusion_reason"] is not None
                        ).items()
                    )
                ),
                # summarize_values: 返回 {min, max, mean, median, p95, ...} 等分布统计
                "original_tokens": summarize_values(
                    [int(row["original_tokens"]) for row in rows]
                ),
                "sequence_tokens": summarize_values(
                    [int(row["sequence_tokens"]) for row in rows]
                ),
                "supervised_tokens": summarize_values(
                    [int(row["supervised_tokens"]) for row in rows]
                ),
                "total_supervised_tokens": sum(
                    int(row["supervised_tokens"]) for row in rows
                ),
                "truncated_count": sum(bool(row["truncated"]) for row in rows),
            }
            for split, rows in audited_by_split.items()
        },
    }

    # ── batch_audit: 对前 4 条合格样本做正确性抽查 ──
    #   样本数量选 4 是为了覆盖: system+user+assistant 的标准对话结构
    #   以及多轮对话的各种变体
    batch_rows = [
        row
        for split in ("train", "validation")
        for row in [
            item
            for item in audited_by_split[split]
            if bool(item["eligible_for_training"])
        ][:4]
    ]
    # 三项关键检查:
    #   1. assistant_mask_nonempty: 每条样本至少有一个 supervised token
    #   2. non_assistant_labels_are_minus_100: 非 assistant token 的 label 都是 -100
    #      并且 assistant token 的 label 等于 token_id 本身（不是 -100）
    #   3. im_end_boundaries_present: 每个 assistant span 后都有 <|im_end|> 边界
    batch_checks = {
        "assistant_mask_nonempty": all(
            int(row["supervised_tokens"]) > 0 for row in batch_rows
        ),
        "non_assistant_labels_are_minus_100": all(
            all(
                label == -100 or label == token_id
                for token_id, label in zip(
                    row["input_ids"], row["labels"], strict=True
                )
            )
            for row in batch_rows
        ),
        "im_end_boundaries_present": all(
            bool(row["im_end_follows_supervised_span"]) for row in batch_rows
        ),
    }
    batch_audit = {
        "status": "passed" if all(batch_checks.values()) else "failed",
        "checks": batch_checks,
        "samples": list(batch_rows),
    }
    return token_report, batch_audit


def run_label_audit(config: Mapping[str, Any]) -> None:
    """编排完整标签审计流程：加载 → 审计 → 报告。"""
    from transformers import AutoTokenizer

    model = config["model"]
    data_dir = Path(config["data"]["output_dir"])

    # ── 第 1 步：加载 tokenizer（无需 BF16 模型，只需 tokenizer）──
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model.get("trust_remote_code", False))
    }
    revision = model.get("tokenizer_revision") or model.get("revision")
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model["id"], **tokenizer_kwargs)
    # 注入训练用聊天模板（含 {% generation %} 标记，确保 mask 行为与训练时一致）
    configure_training_chat_template(tokenizer)

    # ── 第 2 步：读取 train/validation JSONL 文件 ──
    input_paths = {
        split: data_dir / f"{split}.jsonl" for split in ("train", "validation")
    }

    # ── 第 3 步：对每个 split 的每条记录执行 audit_conversation ──
    # read_conversations -> audit_records -> audit_conversation(record) 逐条
    audited = {
        split: audit_records(
            read_conversations(path),
            tokenizer,
            int(config["training"]["max_length"]),
        )
        for split, path in input_paths.items()
    }
    max_length = int(config["training"]["max_length"])

    # ── 第 4 步：构建审计报告 ──
    #   - token_report.json: 全局统计 + 排除列表（供训练阶段过滤用）
    #   - batch_audit.json: 抽查前 4 条合格样本的正确性
    token_report, batch_audit = build_audit_reports(
        audited,
        input_paths,
        model["id"],
        tokenizer,
        max_length,
    )

    # ── 第 5 步：写入数据目录 ──
    write_json(data_dir / "token_report.json", token_report)
    write_json(data_dir / "batch_audit.json", batch_audit)

    # 输出摘要到 stderr
    print(
        "Assistant-only label audit passed: "
        f"train={len(audited['train'])} "
        f"(excluded={token_report['splits']['train']['excluded_count']}), "
        f"validation={len(audited['validation'])} "
        f"(excluded={token_report['splits']['validation']['excluded_count']})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stage 1 Chat Template tokens and labels."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_no_robots.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_stage1_config(config)
    run_label_audit(config)


if __name__ == "__main__":
    main()
