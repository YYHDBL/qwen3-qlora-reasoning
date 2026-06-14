#!/usr/bin/env python3
"""Audit Qwen3 assistant-only labels before any model training."""

from __future__ import annotations

import argparse
import json
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

    original_length = len(input_ids)
    truncated = original_length > max_length
    original_spans = _supervised_spans(mask)
    input_ids = list(input_ids[:max_length])
    attention_mask = list(attention_mask[:max_length])
    mask = mask[:max_length]
    truncated_spans = _supervised_spans(mask)
    labels = [
        token_id if assistant else -100
        for token_id, assistant in zip(input_ids, mask, strict=True)
    ]
    supervised_ids = [
        token_id
        for token_id, label in zip(input_ids, labels, strict=True)
        if label != -100
    ]
    if not supervised_ids:
        raise ValueError(
            f"{record.get('id', '<unknown>')} has no supervised assistant tokens"
        )

    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    ends_with_supervised_token = bool(mask and mask[-1])
    if truncated and ends_with_supervised_token:
        raise ValueError(
            f"{record.get('id', '<unknown>')} cuts through a supervised "
            "assistant span; increase max_length or fix the training "
            "chat template"
        )
    im_end_supervised = any(
        input_ids[end] == im_end_id for _, end in truncated_spans
    )
    if not im_end_supervised:
        raise ValueError(
            f"{record.get('id', '<unknown>')} does not supervise any "
            f"{IM_END_TOKEN} token; check the training chat template "
            "or assistant mask semantics"
        )
    return {
        "id": record.get("id"),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "original_tokens": original_length,
        "sequence_tokens": len(input_ids),
        "supervised_tokens": len(supervised_ids),
        "masked_tokens": len(labels) - len(supervised_ids),
        "truncated": truncated,
        "supervised_span_count": len(truncated_spans),
        "supervised_spans": truncated_spans,
        "original_supervised_span_count": len(original_spans),
        "ends_with_supervised_token": ends_with_supervised_token,
        "im_end_supervised": im_end_supervised,
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    token_report = {
        "model_id": model_id,
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
        row for split in ("train", "validation") for row in audited_by_split[split][:4]
    ]
    batch_audit = {
        "status": "passed",
        "checks": {
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
            "im_end_supervised": all(
                bool(row["im_end_supervised"]) for row in batch_rows
            ),
        },
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
    token_report, batch_audit = build_audit_reports(
        audited, input_paths, model["id"], tokenizer
    )
    write_json(data_dir / "token_report.json", token_report)
    write_json(data_dir / "batch_audit.json", batch_audit)
    print(
        "Assistant-only label audit passed: "
        f"train={len(audited['train'])}, "
        f"validation={len(audited['validation'])}"
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
