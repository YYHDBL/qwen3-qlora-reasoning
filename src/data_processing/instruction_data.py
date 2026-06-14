#!/usr/bin/env python3
"""Prepare and validate the official No Robots conversational splits."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..common.config import load_yaml_config, validate_stage1_config
from ..common.experiment import sha256_file, write_json, write_jsonl


SOURCE_ID = "HuggingFaceH4/no_robots"
SPLIT_NAMES = {"train": "train", "validation": "test"}
ALLOWED_ROLES = {"system", "user", "assistant"}


def validate_conversation_record(
    record: Mapping[str, Any], split: str, index: int
) -> None:
    location = f"{split}[{index}]"
    for field in ("prompt_id", "messages", "category"):
        if field not in record:
            raise ValueError(f"{location} missing field: {field}")
    if not isinstance(record["prompt_id"], str) or not record["prompt_id"]:
        raise ValueError(f"{location} prompt_id must be a nonempty string")
    if not isinstance(record["category"], str) or not record["category"]:
        raise ValueError(f"{location} category must be a nonempty string")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location} messages must be a nonempty list")

    expected = "system_or_user"
    assistant_count = 0
    for message_index, message in enumerate(messages):
        message_location = f"{location}.messages[{message_index}]"
        if not isinstance(message, Mapping):
            raise ValueError(f"{message_location} must be an object")
        if set(message) != {"role", "content"}:
            raise ValueError(f"{message_location} must contain only role and content")
        role = message["role"]
        content = message["content"]
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{message_location} unsupported role: {role}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"{message_location} content must be a nonempty string")
        if message_index == 0 and role == "system":
            expected = "user"
            continue
        if expected == "system_or_user":
            expected = "user"
        if role != expected:
            raise ValueError(
                f"{location} invalid role order at message {message_index}: "
                f"expected {expected}, got {role}"
            )
        if role == "assistant":
            assistant_count += 1
            expected = "user"
        else:
            expected = "assistant"

    if messages[-1]["role"] != "assistant" or assistant_count == 0:
        raise ValueError(f"{location} role order must end with an assistant message")


def prepare_no_robots_records(
    rows: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unsupported Stage 1 split: {split}")
    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        validate_conversation_record(row, split, index)
        record_id = str(row["prompt_id"])
        if record_id in seen_ids:
            raise ValueError(f"{split} contains duplicate id: {record_id}")
        seen_ids.add(record_id)
        prepared.append(
            {
                "id": record_id,
                "messages": copy.deepcopy(row["messages"]),
                "source": SOURCE_ID,
                "category": row["category"],
                "source_split": SPLIT_NAMES[split],
            }
        )
    if not prepared:
        raise ValueError(f"{split} contains no records")
    return prepared


def write_stage1_dataset(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dataset_revision: str | None,
    split_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    train = prepare_no_robots_records(train_rows, "train")
    validation = prepare_no_robots_records(validation_rows, "validation")
    overlap = {row["id"] for row in train} & {row["id"] for row in validation}
    if overlap:
        raise ValueError(f"train and validation IDs overlap: {sorted(overlap)[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": train, "validation": validation}
    split_paths = {split: output_dir / f"{split}.jsonl" for split in split_rows}
    for split, rows in split_rows.items():
        write_jsonl(split_paths[split], rows)

    manifest = {
        "dataset_id": SOURCE_ID,
        "dataset_revision": dataset_revision,
        "splits": {
            split: {
                "source_split": SPLIT_NAMES[split],
                "count": len(rows),
                "sha256": sha256_file(split_paths[split]),
                "ids": [row["id"] for row in rows],
                "arrow_fingerprint": (
                    split_fingerprints.get(split) if split_fingerprints else None
                ),
            }
            for split, rows in split_rows.items()
        },
    }
    report = {
        "status": "ok",
        "total": sum(len(rows) for rows in split_rows.values()),
        "splits": {
            split: {
                "count": len(rows),
                "categories": dict(
                    sorted(Counter(row["category"] for row in rows).items())
                ),
                "message_count": {
                    "min": min(len(row["messages"]) for row in rows),
                    "max": max(len(row["messages"]) for row in rows),
                },
            }
            for split, rows in split_rows.items()
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "dataset_report.json", report)
    return manifest


def prepare_from_hub(config: Mapping[str, Any]) -> dict[str, Any]:
    """Download only the dataset and write validated Stage 1 artifacts."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    data_config = config["data"]
    resolved_revision = data_config.get("dataset_revision")
    if not resolved_revision:
        resolved_revision = HfApi().dataset_info(data_config["dataset_id"]).sha
    kwargs: dict[str, Any] = {"revision": resolved_revision}
    dataset = load_dataset(data_config["dataset_id"], **kwargs)
    train_split = data_config["train_split"]
    validation_split = data_config["validation_split"]
    return write_stage1_dataset(
        list(dataset[train_split]),
        list(dataset[validation_split]),
        Path(data_config["output_dir"]),
        dataset_revision=resolved_revision,
        split_fingerprints={
            "train": dataset[train_split]._fingerprint,
            "validation": dataset[validation_split]._fingerprint,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate Stage 1 No Robots data."
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
    manifest = prepare_from_hub(config)
    print(
        "Prepared Stage 1 data: "
        f"train={manifest['splits']['train']['count']}, "
        f"validation={manifest['splits']['validation']['count']}"
    )


if __name__ == "__main__":
    main()
