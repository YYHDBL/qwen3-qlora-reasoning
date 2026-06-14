import json
from pathlib import Path

import pytest

from src.data_processing.instruction_data import (
    describe_download_environment,
    prepare_no_robots_records,
    validate_download_environment,
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


def test_conversation_validation_accepts_consecutive_assistant_messages():
    value = record(
        messages=[
            {"role": "system", "content": "Stay in character."},
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": "First assistant turn."},
            {"role": "assistant", "content": "Second assistant turn."},
        ]
    )

    validate_conversation_record(value, "train", 487)


def test_conversation_validation_accepts_empty_leading_system_message():
    value = record(
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "Question."},
            {"role": "assistant", "content": "Answer."},
        ]
    )

    validate_conversation_record(value, "train", 7077)


def test_conversation_validation_rejects_bad_role_order():
    value = record(
        messages=[
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Hi."},
        ]
    )

    with pytest.raises(ValueError, match="role order"):
        validate_conversation_record(value, "train", 0)


def test_conversation_validation_rejects_consecutive_user_messages():
    value = record(
        messages=[
            {"role": "user", "content": "First."},
            {"role": "user", "content": "Second."},
            {"role": "assistant", "content": "Reply."},
        ]
    )

    with pytest.raises(ValueError, match="consecutive user"):
        validate_conversation_record(value, "train", 0)


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_conversation_validation_rejects_empty_trainable_messages(role):
    messages = [
        {"role": "user", "content": "Question."},
        {"role": "assistant", "content": "Answer."},
    ]
    messages[0 if role == "user" else 1]["content"] = ""

    with pytest.raises(ValueError, match="content must be a nonempty string"):
        validate_conversation_record(record(messages=messages), "train", 0)


def test_preparation_preserves_messages_and_adds_metadata():
    source = record()
    original_messages = json.loads(json.dumps(source["messages"]))

    prepared = prepare_no_robots_records([source], split="train")

    assert source["messages"] == original_messages
    assert prepared == [
        {
            "id": "train-00000",
            "prompt_id": "p1",
            "messages": original_messages,
            "source": "HuggingFaceH4/no_robots",
            "category": "generation",
            "source_split": "train",
            "source_row_index": 0,
        }
    ]


def test_preparation_keeps_duplicate_prompt_ids_as_unique_records():
    rows = [record("shared"), record("shared")]

    prepared = prepare_no_robots_records(rows, split="train")

    assert [row["id"] for row in prepared] == [
        "train-00000",
        "train-00001",
    ]
    assert [row["prompt_id"] for row in prepared] == ["shared", "shared"]
    assert [row["source_row_index"] for row in prepared] == [0, 1]


def test_preparation_reports_validation_progress():
    messages = []

    prepare_no_robots_records(
        [record("one"), record("two")],
        split="train",
        status=messages.append,
    )

    assert messages == ["Validated train: 2/2 records"]


def test_stage1_writer_is_deterministic_and_maps_official_splits(tmp_path):
    train = [
        record("train-1"),
        record(
            "train-2",
            messages=[
                {"role": "user", "content": "Question."},
                {"role": "assistant", "content": "First."},
                {"role": "assistant", "content": "Second."},
            ],
        ),
    ]
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
    report = json.loads(
        (tmp_path / "one" / "dataset_report.json").read_text(encoding="utf-8")
    )
    assert report["splits"]["train"]["consecutive_assistant_records"] == 1
    assert report["splits"]["train"]["duplicate_prompt_id_records"] == 0
    assert report["splits"]["train"]["empty_leading_system_records"] == 0


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("HF_ENDPOINT", "hf-mirror.com"),
        ("HTTPS_PROXY", "127.0.0.1:7890"),
        ("all_proxy", "localhost:1080"),
    ],
)
def test_download_environment_rejects_urls_without_protocol(variable, value):
    with pytest.raises(ValueError, match=variable):
        validate_download_environment({variable: value})


def test_download_environment_accepts_mirror_and_proxy_urls():
    validate_download_environment(
        {
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HTTPS_PROXY": "http://127.0.0.1:7890",
        }
    )


def test_download_environment_description_hides_proxy_credentials():
    description = describe_download_environment(
        {
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HOME": "/cache/huggingface",
            "HTTPS_PROXY": "http://user:secret@127.0.0.1:7890",
        }
    )

    assert description["endpoint"] == "https://hf-mirror.com"
    assert description["cache"] == "/cache/huggingface"
    assert description["proxies"] == {"HTTPS_PROXY": "http://127.0.0.1:7890"}
