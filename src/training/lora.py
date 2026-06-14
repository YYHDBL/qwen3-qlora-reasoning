"""LoRA configuration construction with delayed PEFT imports."""

from __future__ import annotations

from typing import Any, Mapping


def build_lora_config(config: Mapping[str, Any]) -> Any:
    from peft import LoraConfig, TaskType

    value = config["lora"]
    task_type = getattr(TaskType, str(value["task_type"]))
    return LoraConfig(
        r=int(value["r"]),
        lora_alpha=int(value["lora_alpha"]),
        lora_dropout=float(value["lora_dropout"]),
        target_modules=value["target_modules"],
        bias=str(value["bias"]),
        task_type=task_type,
    )
