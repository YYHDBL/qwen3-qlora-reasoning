import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.evaluate import (
    EvaluationConfig,
    evaluate_records,
    run_evaluation,
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
        "import sys; import src.evaluation.evaluate; "
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


def test_huggingface_generator_uses_transformers_v5_dtype_argument():
    source = Path("src/evaluation/evaluate.py").read_text(encoding="utf-8")

    assert 'model_kwargs["dtype"] = torch.bfloat16' in source
    assert 'model_kwargs["torch_dtype"] = torch.bfloat16' not in source


def test_lora_evaluation_prefers_adapter_tokenizer_for_template_consistency():
    source = Path("src/evaluation/evaluate.py").read_text(encoding="utf-8")

    assert "AutoTokenizer.from_pretrained(\n                    config.adapter_path" in source
    assert "(adapter_dir / \"tokenizer_config.json\").is_file()" in source


def test_evaluate_records_reports_batch_progress():
    records = [
        make_record("one", "cipher", "one"),
        make_record("two", "cipher", "two"),
        make_record("three", "cipher", "three"),
    ]
    progress = []

    evaluate_records(
        records,
        lambda prompts: [
            prompt.removeprefix("Prompt for ").removesuffix("\n\nAnswer:")
            for prompt in prompts
        ],
        batch_size=2,
        progress=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert progress == [(2, 3), (3, 3)]


def test_run_evaluation_reports_major_stages(tmp_path, monkeypatch):
    data_path = tmp_path / "validation.jsonl"
    data_path.write_text(
        json.dumps(make_record("one", "cipher", "one")) + "\n",
        encoding="utf-8",
    )
    messages = []

    class FakeGenerator:
        def __init__(self, config, status):
            status("Loading tokenizer: fake")
            status("Loading model: fake")

        def __call__(self, prompts):
            return ["one" for _ in prompts]

    monkeypatch.setattr(
        "src.evaluation.evaluate.HuggingFaceGenerator", FakeGenerator
    )
    config = EvaluationConfig(
        model_id="Qwen/Qwen3-4B-Base",
        model_mode="bf16",
        split="validation",
        data_path=str(data_path),
        output_dir=str(tmp_path / "output"),
    )

    run_evaluation(config, status=messages.append)

    assert messages[0].startswith("Starting evaluation:")
    assert any(message.startswith("Loading dataset:") for message in messages)
    assert "Loaded 1 validation records" in messages
    assert "Loading tokenizer: fake" in messages
    assert "Loading model: fake" in messages
    assert "Generated 1/1 samples (100.0%)" in messages
    assert any(
        message.startswith("Writing evaluation artifacts:")
        for message in messages
    )
    assert messages[-1].startswith("Evaluation complete:")
