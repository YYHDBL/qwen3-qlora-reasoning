import subprocess
import sys
from pathlib import Path

import pytest

from src.training.train_sft import (
    build_sft_kwargs,
    load_audit_exclusions,
    resolve_run_limits,
    snapshot_data_artifacts,
    validate_preflight_artifacts,
)
from src.training.model_loader import trim_generated_ids


def test_training_modules_import_without_gpu_dependencies():
    code = (
        "import sys; "
        "import src.training.train_sft; "
        "import src.training.model_loader; "
        "import src.training.verify_adapter; "
        "assert 'torch' not in sys.modules; "
        "assert 'datasets' not in sys.modules; "
        "assert 'peft' not in sys.modules; "
        "assert 'trl' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_run_modes_apply_expected_limits():
    training = {
        "max_steps": 100,
        "num_train_epochs": 1,
        "overfit_examples": 16,
        "overfit_steps": 40,
        "smoke_examples": 256,
        "smoke_steps": 20,
    }

    assert resolve_run_limits("overfit", training) == {
        "example_limit": 16,
        "max_steps": 40,
        "num_train_epochs": None,
    }
    assert resolve_run_limits("smoke", training) == {
        "example_limit": 256,
        "max_steps": 20,
        "num_train_epochs": None,
    }
    assert resolve_run_limits("formal", training) == {
        "example_limit": None,
        "max_steps": 100,
        "num_train_epochs": 1,
    }


def test_sft_kwargs_enable_bf16_lora_training_contract(tmp_path):
    config = {
        "training": {
            "max_length": 2048,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            "assistant_only_loss": True,
            "packing": False,
            "learning_rate": 0.0001,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch_fused",
            "logging_steps": 1,
            "save_steps": 10,
            "eval_steps": 10,
            "seed": 42,
            "bf16": True,
            "tf32": True,
        }
    }

    kwargs = build_sft_kwargs(config, output_dir=tmp_path, run_mode="smoke")

    assert kwargs["assistant_only_loss"] is True
    assert kwargs["bf16"] is True
    assert kwargs["max_length"] == 2048
    assert kwargs["max_steps"] == 20
    assert kwargs["report_to"] == ["tensorboard"]


def test_formal_mode_requires_all_preflight_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="preflight"):
        validate_preflight_artifacts(tmp_path)

    for name in (
        "dataset_manifest.json",
        "token_report.json",
        "batch_audit.json",
        "overfit_passed.json",
        "adapter_reload.json",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    validate_preflight_artifacts(tmp_path)


def test_generated_ids_stop_at_first_chat_terminator_before_batch_padding():
    token_ids, stop_reason = trim_generated_ids(
        [7, 8, 99, 0, 0],
        {"im_end": 99, "endoftext": 0},
    )

    assert token_ids == [7, 8, 99]
    assert stop_reason == "im_end"


def test_training_run_snapshots_data_audit_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "run"
    data_dir.mkdir()
    for name in (
        "dataset_manifest.json",
        "token_report.json",
        "batch_audit.json",
    ):
        (data_dir / name).write_text('{"status":"ok"}\n', encoding="utf-8")

    snapshot_data_artifacts({"data": {"output_dir": str(data_dir)}}, output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "dataset_manifest.json",
        "token_report.json",
        "batch_audit.json",
    }


def test_training_loads_split_exclusions_from_matching_token_report(tmp_path):
    report_path = tmp_path / "token_report.json"
    report_path.write_text(
        """
{
  "max_length": 2048,
  "splits": {
    "train": {"excluded_ids": ["train-00612"]},
    "validation": {"excluded_ids": ["test-00007"]}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exclusions = load_audit_exclusions(tmp_path, expected_max_length=2048)

    assert exclusions == {
        "train": {"train-00612"},
        "validation": {"test-00007"},
    }


def test_training_rejects_token_report_for_different_max_length(tmp_path):
    (tmp_path / "token_report.json").write_text(
        '{"max_length": 4096, "splits": {}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_length"):
        load_audit_exclusions(tmp_path, expected_max_length=2048)
