import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.analyze_tokens import (
    analyze_dataset,
    analyze_records,
    build_report,
    collect_tokenizer_metadata,
    load_datasets,
    percentile,
    read_jsonl,
    render_markdown_report,
)
from src.common.prompt_format import format_training_text


REQUIRED_FIELDS = {"id", "task_type", "prompt", "answer"}


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token = None
    pad_token_id = None
    bos_token = None
    bos_token_id = None
    vocab_size = 256
    init_kwargs = {"_commit_hash": "fake-revision"}

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": list(text.encode("utf-8"))}


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def make_record(record_id: str, task_type: str = "cipher") -> dict[str, str]:
    return {
        "id": record_id,
        "task_type": task_type,
        "prompt": "Solve this.",
        "answer": "done",
    }


def make_data_dir(tmp_path: Path) -> Path:
    records_by_split = {
        "train": [make_record("train-1"), make_record("train-2")],
        "validation": [make_record("validation-1")],
        "test": [make_record("test-1")],
    }
    for split, records in records_by_split.items():
        write_jsonl(tmp_path / f"{split}.jsonl", records)
    return tmp_path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_datasets_reads_all_three_split_counts(tmp_path):
    data_dir = make_data_dir(tmp_path)

    records, input_files = load_datasets(data_dir)

    assert len(records) == 4
    assert {split: sum(r["split"] == split for r in records) for split in (
        "train",
        "validation",
        "test",
    )} == {"train": 2, "validation": 1, "test": 1}
    assert set(input_files) == {"train", "validation", "test"}


def test_format_training_text_uses_exact_template_and_eos():
    prompt = "Question?"
    answer = "42"

    prompt_text, answer_text, full_text = format_training_text(
        prompt, answer, "<eos>"
    )

    assert prompt_text == "Question?\n\nAnswer:\n"
    assert answer_text == "42<eos>"
    assert full_text == "Question?\n\nAnswer:\n42<eos>"


def test_format_training_text_does_not_modify_original_values():
    prompt = "  Keep prompt whitespace  "
    answer = " keep answer whitespace "
    original_prompt = prompt
    original_answer = answer

    format_training_text(prompt, answer, "<eos>")

    assert prompt == original_prompt
    assert answer == original_answer


def test_percentile_uses_linear_interpolation():
    values = [1, 2, 3, 4]

    assert percentile(values, 0) == 1
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)
    assert percentile(values, 1) == 4


def test_read_jsonl_rejects_empty_data(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no records"):
        read_jsonl(path, "train")


def test_read_jsonl_rejects_missing_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"id": "one", "prompt": "p", "answer": "a"}])

    with pytest.raises(ValueError, match="missing required fields: task_type"):
        read_jsonl(path, "train")


def test_repeated_analysis_has_identical_core_statistics(tmp_path):
    data_dir = make_data_dir(tmp_path)
    records, input_files = load_datasets(data_dir)
    tokenizer = FakeTokenizer()

    analyzed_once = analyze_records(records, tokenizer)
    analyzed_twice = analyze_records(records, tokenizer)
    report_once = build_report(
        analyzed_once,
        tokenizer,
        "fake/model",
        input_files,
        execution_time="2026-06-13T00:00:00Z",
    )
    report_twice = build_report(
        analyzed_twice,
        tokenizer,
        "fake/model",
        input_files,
        execution_time="2026-06-13T00:00:00Z",
    )

    for key in ("statistics", "overflow", "longest_samples", "recommendation"):
        assert report_once[key] == report_twice[key]


def test_report_is_json_serializable(tmp_path):
    data_dir = make_data_dir(tmp_path)
    records, input_files = load_datasets(data_dir)
    tokenizer = FakeTokenizer()
    analyzed = analyze_records(records, tokenizer)
    report = build_report(
        analyzed,
        tokenizer,
        "fake/model",
        input_files,
        execution_time="2026-06-13T00:00:00Z",
    )

    serialized = json.dumps(report)

    assert serialized
    assert "# Tokenizer Length Analysis" in render_markdown_report(report)


def test_analyze_dataset_preserves_all_input_hashes(tmp_path):
    data_dir = make_data_dir(tmp_path)
    paths = {
        split: data_dir / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    hashes_before = {split: sha256(path) for split, path in paths.items()}

    report = analyze_dataset(
        data_dir=data_dir,
        tokenizer=FakeTokenizer(),
        model_id="fake/model",
        execution_time="2026-06-13T00:00:00Z",
    )

    hashes_after = {split: sha256(path) for split, path in paths.items()}
    assert hashes_after == hashes_before
    assert all(
        report["input_files"][split]["unchanged"]
        for split in ("train", "validation", "test")
    )


def test_record_shape_is_preserved_during_token_analysis():
    source = make_record("record-1")
    source_before = dict(source)
    source["split"] = "train"

    analyzed = analyze_records([source], FakeTokenizer())

    assert {key: source[key] for key in REQUIRED_FIELDS} == source_before
    assert analyzed[0]["id"] == "record-1"
    assert analyzed[0]["prompt_tokens"] > 0
    assert analyzed[0]["answer_tokens"] > 0
    assert analyzed[0]["full_sequence_tokens"] > 0


def test_tokenizer_metadata_uses_cached_revision_when_available():
    tokenizer = FakeTokenizer()
    tokenizer.init_kwargs = {}
    tokenizer._analysis_revision = "cached-commit"

    metadata = collect_tokenizer_metadata(tokenizer, "fake/model")

    assert metadata["revision"] == "cached-commit"
