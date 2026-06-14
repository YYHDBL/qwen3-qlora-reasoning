#!/usr/bin/env python3
"""Audit Qwen3 assistant-only labels before any model training."""

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
    """Return whether every full assistant span has an <|im_end|> terminator."""
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
    rendered = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    input_ids = rendered["input_ids"]
    attention_mask = rendered.get("attention_mask", [1] * len(input_ids))
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
    mask = _assistant_mask(rendered)
    if not (len(input_ids) == len(attention_mask) == len(mask)):
        raise ValueError("tokenizer returned inconsistent mask lengths")

    original_input_ids = list(input_ids)
    original_attention_mask = list(attention_mask)
    original_mask = list(mask)
    original_length = len(original_input_ids)
    original_spans = _supervised_spans(original_mask)
    original_supervised_tokens = sum(bool(value) for value in original_mask)
    if not original_supervised_tokens:
        raise ValueError(
            f"{record.get('id', '<unknown>')} has no supervised assistant tokens"
        )

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

    truncated = original_length > max_length
    eligible_for_training = not truncated
    exclusion_reason = "sequence_exceeds_max_length" if truncated else None
    truncates_supervised_span = any(
        start < max_length <= end for start, end in original_spans
    )
    input_ids = original_input_ids[:max_length]
    attention_mask = original_attention_mask[:max_length]
    mask = original_mask[:max_length]
    retained_spans = _supervised_spans(mask)
    labels = [
        token_id if assistant else -100
        for token_id, assistant in zip(input_ids, mask, strict=True)
    ]
    supervised_ids = [
        token_id
        for token_id, label in zip(input_ids, labels, strict=True)
        if label != -100
    ]

    ends_with_supervised_token = bool(mask and mask[-1])
    im_end_supervised = any(
        token_id == im_end_id and assistant
        for token_id, assistant in zip(input_ids, mask, strict=True)
    )
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
    return [audit_conversation(record, tokenizer, max_length) for record in records]


def build_audit_reports(
    audited_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    input_paths: Mapping[str, Path],
    model_id: str,
    tokenizer: Any,
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
                "excluded_ids": [
                    row["id"]
                    for row in rows
                    if not bool(row["eligible_for_training"])
                ],
                "exclusion_reasons": dict(
                    sorted(
                        Counter(
                            str(row["exclusion_reason"])
                            for row in rows
                            if row["exclusion_reason"] is not None
                        ).items()
                    )
                ),
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
    batch_rows = [
        row
        for split in ("train", "validation")
        for row in [
            item
            for item in audited_by_split[split]
            if bool(item["eligible_for_training"])
        ][:4]
    ]
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
    from transformers import AutoTokenizer

    model = config["model"]
    data_dir = Path(config["data"]["output_dir"])
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model.get("trust_remote_code", False))
    }
    revision = model.get("tokenizer_revision") or model.get("revision")
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model["id"], **tokenizer_kwargs)
    configure_training_chat_template(tokenizer)

    input_paths = {
        split: data_dir / f"{split}.jsonl" for split in ("train", "validation")
    }
    audited = {
        split: audit_records(
            read_conversations(path),
            tokenizer,
            int(config["training"]["max_length"]),
        )
        for split, path in input_paths.items()
    }
    max_length = int(config["training"]["max_length"])
    token_report, batch_audit = build_audit_reports(
        audited,
        input_paths,
        model["id"],
        tokenizer,
        max_length,
    )
    write_json(data_dir / "token_report.json", token_report)
    write_json(data_dir / "batch_audit.json", batch_audit)
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
