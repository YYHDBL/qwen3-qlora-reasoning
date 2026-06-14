#!/usr/bin/env python3
"""Prepare and validate the official No Robots conversational splits."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from ..common.config import load_yaml_config, validate_stage1_config
from ..common.experiment import sha256_file, write_json, write_jsonl


SOURCE_ID = "HuggingFaceH4/no_robots"
SPLIT_NAMES = {"train": "train", "validation": "test"}
ALLOWED_ROLES = {"system", "user", "assistant"}
NETWORK_URL_VARIABLES = (
    "HF_ENDPOINT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
StatusCallback = Callable[[str], None]


def console_status(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(
        f"[{timestamp}] [instruction-data] {message}",
        file=sys.stderr,
        flush=True,
    )


def validate_download_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject malformed Hub endpoint and proxy URLs before HTTPX sees them."""
    values = environment if environment is not None else os.environ
    for variable in NETWORK_URL_VARIABLES:
        value = values.get(variable)
        if not value:
            continue
        parsed = urlsplit(value)
        allowed_schemes = (
            {"http", "https"}
            if variable == "HF_ENDPOINT"
            else {"http", "https", "socks5", "socks5h"}
        )
        if parsed.scheme.lower() not in allowed_schemes or not parsed.netloc:
            examples = (
                "https://hf-mirror.com"
                if variable == "HF_ENDPOINT"
                else "http://127.0.0.1:7890"
            )
            raise ValueError(
                f"{variable} must be a complete URL including protocol, "
                f"for example {examples}; got {value!r}"
            )


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def describe_download_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = environment if environment is not None else os.environ
    return {
        "endpoint": values.get("HF_ENDPOINT", "https://huggingface.co"),
        "cache": values.get("HF_HOME", "<huggingface default>"),
        "proxies": {
            variable: _redact_url(values[variable])
            for variable in NETWORK_URL_VARIABLES
            if variable != "HF_ENDPOINT" and values.get(variable)
        },
    }


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

    previous_role: str | None = None
    assistant_count = 0
    user_count = 0
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
        if not isinstance(content, str):
            raise ValueError(f"{message_location} content must be a string")
        if not content and not (message_index == 0 and role == "system"):
            raise ValueError(f"{message_location} content must be a nonempty string")
        if role == "system":
            if message_index != 0:
                raise ValueError(f"{location} system role is only allowed at message 0")
            previous_role = role
            continue
        if message_index == 0 and role != "user":
            raise ValueError(
                f"{location} invalid role order at message {message_index}: "
                f"expected user, got {role}"
            )
        if message_index == 1 and previous_role == "system" and role != "user":
            raise ValueError(
                f"{location} invalid role order at message {message_index}: "
                f"expected user, got {role}"
            )
        if role == "user":
            user_count += 1
            if previous_role == "user":
                raise ValueError(
                    f"{location} contains consecutive user messages at "
                    f"message {message_index}"
                )
        if role == "assistant":
            assistant_count += 1
        previous_role = role

    if user_count == 0 or assistant_count == 0:
        raise ValueError(f"{location} must contain user and assistant messages")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{location} role order must end with an assistant message")


def prepare_no_robots_records(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    status: StatusCallback | None = None,
) -> list[dict[str, Any]]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unsupported Stage 1 split: {split}")
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        validate_conversation_record(row, split, index)
        prepared.append(
            {
                "id": f"{SPLIT_NAMES[split]}-{index:05d}",
                "prompt_id": str(row["prompt_id"]),
                "messages": copy.deepcopy(row["messages"]),
                "source": SOURCE_ID,
                "category": row["category"],
                "source_split": SPLIT_NAMES[split],
                "source_row_index": index,
            }
        )
        completed = index + 1
        if status is not None and (completed % 1000 == 0 or completed == len(rows)):
            status(f"Validated {split}: {completed}/{len(rows)} records")
    if not prepared:
        raise ValueError(f"{split} contains no records")
    return prepared


def write_stage1_dataset(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dataset_revision: str | None,
    split_fingerprints: Mapping[str, str] | None = None,
    status: StatusCallback | None = None,
) -> dict[str, Any]:
    train = prepare_no_robots_records(train_rows, "train", status=status)
    validation = prepare_no_robots_records(validation_rows, "validation", status=status)
    overlap = {row["id"] for row in train} & {row["id"] for row in validation}
    if overlap:
        raise ValueError(f"train and validation IDs overlap: {sorted(overlap)[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": train, "validation": validation}
    split_paths = {split: output_dir / f"{split}.jsonl" for split in split_rows}
    for split, rows in split_rows.items():
        if status is not None:
            status(f"Writing {split} JSONL: {split_paths[split]}")
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
                "consecutive_assistant_records": sum(
                    any(
                        first["role"] == second["role"] == "assistant"
                        for first, second in zip(
                            row["messages"],
                            row["messages"][1:],
                            strict=False,
                        )
                    )
                    for row in rows
                ),
                "duplicate_prompt_id_records": sum(
                    count
                    for count in Counter(row["prompt_id"] for row in rows).values()
                    if count > 1
                ),
                "duplicate_prompt_id_values": sum(
                    count > 1
                    for count in Counter(row["prompt_id"] for row in rows).values()
                ),
                "empty_leading_system_records": sum(
                    bool(row["messages"])
                    and row["messages"][0]["role"] == "system"
                    and row["messages"][0]["content"] == ""
                    for row in rows
                ),
            }
            for split, rows in split_rows.items()
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "dataset_report.json", report)
    if status is not None:
        status(f"Wrote dataset manifest and report: {output_dir}")
    return manifest


def prepare_from_hub(
    config: Mapping[str, Any],
    status: StatusCallback = console_status,
) -> dict[str, Any]:
    """Download only the dataset and write validated Stage 1 artifacts."""
    validate_download_environment()
    environment = describe_download_environment()
    status(f"Hub endpoint: {environment['endpoint']}; cache: {environment['cache']}")
    if environment["proxies"]:
        status(f"Proxy configuration: {environment['proxies']}")

    from datasets import enable_progress_bars, load_dataset
    from huggingface_hub import HfApi
    from huggingface_hub.utils import enable_progress_bars as enable_hub_progress

    enable_progress_bars()
    enable_hub_progress()

    data_config = config["data"]
    resolved_revision = data_config.get("dataset_revision")
    if not resolved_revision:
        status(f"Resolving dataset revision: {data_config['dataset_id']}")
        resolved_revision = HfApi().dataset_info(data_config["dataset_id"]).sha
    status(
        f"Loading dataset {data_config['dataset_id']} at revision {resolved_revision}"
    )
    kwargs: dict[str, Any] = {"revision": resolved_revision}
    dataset = load_dataset(data_config["dataset_id"], **kwargs)
    train_split = data_config["train_split"]
    validation_split = data_config["validation_split"]
    status(
        "Dataset loaded: "
        f"{train_split}={len(dataset[train_split])}, "
        f"{validation_split}={len(dataset[validation_split])}"
    )
    return write_stage1_dataset(
        list(dataset[train_split]),
        list(dataset[validation_split]),
        Path(data_config["output_dir"]),
        dataset_revision=resolved_revision,
        split_fingerprints={
            "train": dataset[train_split]._fingerprint,
            "validation": dataset[validation_split]._fingerprint,
        },
        status=status,
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
