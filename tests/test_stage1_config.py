from pathlib import Path

import pytest

from src.common.config import (
    apply_overrides,
    load_yaml_config,
    validate_stage1_config,
)
from src.common.experiment import ensure_new_output_dir


def test_stage1_config_loads_and_validates():
    config = load_yaml_config(Path("configs/stage1_no_robots.yaml"))

    validate_stage1_config(config)

    assert config["model"]["id"] == "Qwen/Qwen3-4B-Base"
    assert config["training"]["bf16"] is True
    assert config["training"]["assistant_only_loss"] is True
    assert config["lora"]["target_modules"] == "all-linear"


def test_config_override_updates_nested_value_without_mutating_source():
    source = {"training": {"max_steps": 20}, "model": {"id": "base"}}

    updated = apply_overrides(source, ["training.max_steps=3"])

    assert source["training"]["max_steps"] == 20
    assert updated["training"]["max_steps"] == 3


def test_existing_output_directory_requires_resume(tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_new_output_dir(output_dir)


def test_invalid_stage1_config_rejects_quantization():
    config = load_yaml_config(Path("configs/stage1_no_robots.yaml"))
    config["model"]["load_in_4bit"] = True

    with pytest.raises(ValueError, match="4-bit"):
        validate_stage1_config(config)
