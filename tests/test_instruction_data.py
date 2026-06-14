import json
from pathlib import Path

import pytest

from src.data_processing.instruction_data import (
    prepare_no_robots_records,
    validate_conversation_record,
    write_stage1_dataset,
)


def record(record_id="p1", messages=None):
    return {
        "prompt": "ignored display prompt",
        "prompt_id": record_id,
        "messages": messages
        or [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
        ],
        "category": "generation",
    }


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_conversation_validation_accepts_system_and_multiturn():
    value = record(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "One?"},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": "Two?"},
            {"role": "assistant", "content": "2"},
        ]
    )

    validate_conversation_record(value, "train", 0)


def test_conversation_validation_rejects_bad_role_order():
    value = record(
        messages=[
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Hi."},
        ]
    )

    with pytest.raises(ValueError, match="role order"):
        validate_conversation_record(value, "train", 0)


def test_preparation_preserves_messages_and_adds_metadata():
    source = record()
    original_messages = json.loads(json.dumps(source["messages"]))

    prepared = prepare_no_robots_records([source], split="train")

    assert source["messages"] == original_messages
    assert prepared == [
        {
            "id": "p1",
            "messages": original_messages,
            "source": "HuggingFaceH4/no_robots",
            "category": "generation",
            "source_split": "train",
        }
    ]


def test_stage1_writer_is_deterministic_and_maps_official_splits(tmp_path):
    train = [record("train-1"), record("train-2")]
    validation = [record("valid-1")]

    manifest_one = write_stage1_dataset(
        train, validation, tmp_path / "one", dataset_revision="abc"
    )
    manifest_two = write_stage1_dataset(
        train, validation, tmp_path / "two", dataset_revision="abc"
    )

    assert (
        manifest_one["splits"]["train"]["sha256"]
        == (manifest_two["splits"]["train"]["sha256"])
    )
    assert (
        manifest_one["splits"]["validation"]["sha256"]
        == (manifest_two["splits"]["validation"]["sha256"])
    )
    assert read_jsonl(tmp_path / "one" / "train.jsonl")[0]["source_split"] == "train"
    assert (
        read_jsonl(tmp_path / "one" / "validation.jsonl")[0]["source_split"] == "test"
    )
