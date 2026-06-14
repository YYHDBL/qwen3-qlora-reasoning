"""可重现实验的 YAML 配置辅助函数。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    """从磁盘加载一个 YAML 映射。"""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return value


def apply_overrides(config: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    """返回深拷贝并应用点号分隔的 ``key=value`` 覆盖项。"""
    updated = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value syntax: {override}")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        if not all(keys):
            raise ValueError(f"invalid override key: {dotted_key}")
        target: dict[str, Any] = updated
        for key in keys[:-1]:
            child = target.get(key)
            if not isinstance(child, dict):
                raise KeyError(f"override path does not exist: {dotted_key}")
            target = child
        if keys[-1] not in target:
            raise KeyError(f"override key does not exist: {dotted_key}")
        target[keys[-1]] = yaml.safe_load(raw_value)
    return updated


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section must be a mapping: {key}")
    return value


def validate_stage1_config(config: Mapping[str, Any]) -> None:
    """校验 BF16 Stage 1 实验所需的约束条件。"""
    experiment = _require_mapping(config, "experiment")
    model = _require_mapping(config, "model")
    data = _require_mapping(config, "data")
    training = _require_mapping(config, "training")
    lora = _require_mapping(config, "lora")
    evaluation = _require_mapping(config, "evaluation")

    if not experiment.get("swanlab_project") or not isinstance(experiment.get("swanlab_project"), str):
        raise ValueError("experiment.swanlab_project is required and must be a non-empty string")
    if model.get("id") != "Qwen/Qwen3-4B-Base":
        raise ValueError("Stage 1 model must be Qwen/Qwen3-4B-Base")
    if model.get("load_in_4bit") or model.get("load_in_8bit"):
        raise ValueError("Stage 1 forbids 4-bit and 8-bit model loading")
    if data.get("dataset_id") != "HuggingFaceH4/no_robots":
        raise ValueError("Stage 1 dataset must be HuggingFaceH4/no_robots")
    if training.get("bf16") is not True:
        raise ValueError("Stage 1 requires BF16 training")
    if training.get("assistant_only_loss") is not True:
        raise ValueError("Stage 1 requires assistant-only loss")
    if lora.get("target_modules") != "all-linear":
        raise ValueError("Stage 1 LoRA target_modules must be all-linear")
    if int(training.get("max_length", 0)) <= 0:
        raise ValueError("training.max_length must be positive")
    if evaluation.get("dev_path") == evaluation.get("test_path"):
        raise ValueError("instruction dev and test paths must differ")
