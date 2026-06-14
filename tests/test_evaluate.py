import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluate import (
    EvaluationConfig,
    evaluate_records,
    validate_split_access,
    write_evaluation_artifacts,
)


def make_record(record_id: str, task_type: str, answer: str):
    return {
        "id": record_id,
        "task_type": task_type,
        "prompt": f"Prompt for {record_id}",
        "answer": answer,
        "split": "validation",
    }


def test_evaluate_records_reuses_prompt_and_answer_modules():
    records = [
        make_record("one", "bit_manipulation", "10010111"),
        make_record("two", "gravity", "57.0"),
    ]
    seen_prompts = []

    def generate(prompts):
        seen_prompts.extend(prompts)
        return ["10010111", "57.00"]

    predictions = evaluate_records(records, generate, batch_size=2)

    assert seen_prompts == [
        "Prompt for one\n\nAnswer:",
        "Prompt for two\n\nAnswer:",
    ]
    assert predictions[0]["primary_correct"] is True
    assert predictions[1]["primary_correct"] is True
    assert predictions[1]["strict_correct"] is False
    assert "prompt" not in predictions[0]


def test_artifact_writer_creates_required_files(tmp_path):
    predictions = evaluate_records(
        [make_record("one", "cipher", "hello world")],
        lambda prompts: ["wrong"],
        batch_size=1,
    )
    config = EvaluationConfig(
        model_id="Qwen/Qwen3-4B-Base",
        model_mode="bf16",
        split="validation",
        data_path="data/processed/validation.jsonl",
        output_dir=str(tmp_path),
    )

    write_evaluation_artifacts(
        output_dir=tmp_path,
        predictions=predictions,
        run_config=config.to_dict(),
    )

    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "predictions.jsonl",
        "error_cases.jsonl",
        "metrics.json",
        "run_config.json",
    }
    errors = [
        json.loads(line)
        for line in (tmp_path / "error_cases.jsonl").read_text().splitlines()
    ]
    assert [record["id"] for record in errors] == ["one"]
    assert json.loads((tmp_path / "metrics.json").read_text())[
        "overall"
    ]["primary_accuracy"] == 0.0


def test_test_split_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="protected test split"):
        validate_split_access("test", allow_test=False)

    validate_split_access("validation", allow_test=False)
    validate_split_access("test", allow_test=True)


def test_evaluation_module_imports_without_model_dependencies():
    code = (
        "import sys; import src.evaluate; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'peft' not in sys.modules; "
        "assert 'bitsandbytes' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_lora_mode_requires_adapter_path():
    with pytest.raises(ValueError, match="adapter path"):
        EvaluationConfig(
            model_id="Qwen/Qwen3-4B-Base",
            model_mode="lora",
            split="validation",
            data_path="data/processed/validation.jsonl",
            output_dir="outputs/run",
        )
